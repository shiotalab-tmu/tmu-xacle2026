"""
SwiGLU Audio Projection Module

Audio adapter that converts BEATs encoder output to LLM input space.
Uses SwiGLU MLP with optional gated residual connection.

Reference:
    Shazeer, "GLU Variants Improve Transformer" (2020)
    https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioAdapterOutput:
    """Output from audio adapter."""

    audio_tokens: torch.Tensor  # [B, num_tokens, llm_hidden_dim]
    attention_mask: torch.Tensor  # [B, num_tokens]


class SwiGLU(nn.Module):
    """
    SwiGLU activation: Swish(x_gate) * x_value

    Input: [*, 2*d] -> Output: [*, d]
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_value, x_gate = x.chunk(2, dim=-1)
        return x_value * F.silu(x_gate)


class SwiGLUMLP(nn.Module):
    """
    SwiGLU MLP layer.

    Architecture:
        LayerNorm → Linear(d_in → 2*d_hidden) → SwiGLU → Dropout → Linear(d_hidden → d_out)

    Args:
        d_in: Input dimension
        d_hidden: Hidden dimension (after SwiGLU split)
        d_out: Output dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out

        # Layer norm
        self.norm = nn.LayerNorm(d_in)

        # Linear layers
        # First linear projects to 2*d_hidden for SwiGLU split
        self.fc1 = nn.Linear(d_in, 2 * d_hidden)
        self.swiglu = SwiGLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [*, d_in]

        Returns:
            Output tensor [*, d_out]
        """
        x = self.norm(x)
        x = self.fc1(x)  # [*, 2*d_hidden]
        x = self.swiglu(x)  # [*, d_hidden]
        x = self.dropout(x)
        x = self.fc2(x)  # [*, d_out]
        return x


class SwiGLUProjection(nn.Module):
    """
    Audio Projection with SwiGLU MLP and Gated Residual Connection.

    Architecture:
        1. Temporal pooling: [B, T_audio, audio_dim] -> [B, num_tokens, audio_dim]
        2. SwiGLU MLP: [B, num_tokens, audio_dim] -> [B, num_tokens, llm_hidden_dim]
        3. Gated Residual: gate * residual + (1 - gate) * mlp_output

    Paper configuration:
        - audio_dim: 768 (BEATs)
        - num_tokens: 100
        - intermediate_dim: 3584 (4 * 896)
        - llm_hidden_dim: 896 (Qwen2.5-0.5B)
        - dropout: 0.1

    Args:
        audio_dim: Input dimension from audio encoder (768 for BEATs)
        llm_hidden_dim: LLM hidden dimension (896 for Qwen2.5-0.5B)
        num_tokens: Number of output audio tokens (100)
        intermediate_dim: Intermediate dimension, defaults to 4 * llm_hidden_dim
        dropout: Dropout probability (0.1)
        use_residual_gate: Use gated residual connection (True)
    """

    def __init__(
        self,
        audio_dim: int = 768,
        llm_hidden_dim: int = 896,
        num_tokens: int = 100,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_residual_gate: bool = True,
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.num_tokens = num_tokens
        self.intermediate_dim = intermediate_dim or (4 * llm_hidden_dim)
        self.dropout_prob = dropout
        self.use_residual_gate = use_residual_gate

        # Temporal pooling: T_audio -> num_tokens
        self.audio_pooling = nn.AdaptiveAvgPool1d(num_tokens)

        # SwiGLU MLP
        self.mlp = SwiGLUMLP(
            d_in=audio_dim,
            d_hidden=self.intermediate_dim,
            d_out=llm_hidden_dim,
            dropout=dropout,
        )

        # Gated Residual Connection
        if use_residual_gate:
            self.residual_proj = nn.Linear(audio_dim, llm_hidden_dim)
            self.residual_gate = nn.Linear(audio_dim, llm_hidden_dim)
        else:
            self.residual_proj = None
            self.residual_gate = None

        print(f"[SwiGLUProjection] Initialized:")
        print(f"  Audio dim: {audio_dim}")
        print(f"  LLM hidden dim: {llm_hidden_dim}")
        print(f"  Num tokens: {num_tokens}")
        print(f"  Intermediate dim: {self.intermediate_dim}")
        print(f"  Dropout: {dropout}")
        print(f"  Residual gate: {use_residual_gate}")

    def forward(self, audio_latents: torch.Tensor) -> AudioAdapterOutput:
        """
        Project audio features to LLM input space.

        Args:
            audio_latents: [B, T_audio, audio_dim] from BEATs encoder

        Returns:
            AudioAdapterOutput with:
                - audio_tokens: [B, num_tokens, llm_hidden_dim]
                - attention_mask: [B, num_tokens]
        """
        # Handle 2D input (global embeddings)
        if audio_latents.ndim == 2:
            audio_latents = audio_latents.unsqueeze(1).expand(-1, self.num_tokens, -1)

        # Temporal pooling: [B, T_audio, D] -> [B, num_tokens, D]
        # AdaptiveAvgPool1d expects (B, C, L) format
        x = self.audio_pooling(
            audio_latents.transpose(1, 2)
        ).transpose(1, 2)  # [B, num_tokens, audio_dim]

        # SwiGLU MLP
        audio_tokens = self.mlp(x)  # [B, num_tokens, llm_hidden_dim]

        # Gated Residual Connection
        if self.use_residual_gate:
            residual = self.residual_proj(x)  # [B, num_tokens, llm_hidden_dim]
            gate = torch.sigmoid(self.residual_gate(x))  # [B, num_tokens, llm_hidden_dim]
            audio_tokens = gate * residual + (1 - gate) * audio_tokens

        # Create attention mask (all ones)
        B = audio_tokens.size(0)
        attention_mask = torch.ones(
            B, self.num_tokens, device=audio_tokens.device, dtype=torch.long
        )

        return AudioAdapterOutput(
            audio_tokens=audio_tokens,
            attention_mask=attention_mask,
        )
