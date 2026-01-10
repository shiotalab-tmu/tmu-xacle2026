"""
Score Head for Alignment Score Prediction

MLP-based projection head that converts LLM hidden states to alignment scores.
"""

import torch
import torch.nn as nn
from typing import Literal


class ScoreHead(nn.Module):
    """
    MLP Score Head for alignment score prediction.

    Architecture:
        input_dim -> hidden_dim -> hidden_dim -> output_dim

    Paper configuration:
        - input_dim: 896 (Qwen2.5-0.5B hidden_dim)
        - hidden_dim: 512
        - output_dim: 128 (logits dimension for wrapper)

    The final score is computed by the wrapper using a linear layer.

    Args:
        input_dim: Input dimension from LLM (896 for Qwen2.5-0.5B)
        hidden_dim: Hidden layer dimension (512)
        output_dim: Output logits dimension (128)
        activation: Activation function
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 896,
        hidden_dim: int = 512,
        output_dim: int = 128,
        activation: Literal["ReLU", "GELU", "SiLU"] = "ReLU",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Get activation function
        activation_fn = self._get_activation(activation)

        # 3-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

        print(f"[ScoreHead] Initialized: {input_dim} -> {hidden_dim} -> {output_dim}")

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_map = {
            "ReLU": nn.ReLU(),
            "GELU": nn.GELU(),
            "SiLU": nn.SiLU(),
        }
        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}")
        return activation_map[activation]

    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, input_dim] hidden state from [SCORE] token

        Returns:
            [B, output_dim] logits
        """
        return self.mlp(x)


class LinearScoreHead(nn.Module):
    """
    Simple linear score head for final score prediction.

    Maps logits to scalar score.

    Args:
        input_dim: Input dimension (128 from ScoreHead output)
    """

    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, input_dim] logits

        Returns:
            [B] scalar scores
        """
        return self.linear(x).squeeze(-1)
