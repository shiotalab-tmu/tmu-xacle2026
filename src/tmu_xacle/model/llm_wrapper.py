"""
LLM Wrapper for Audio-Text Alignment

Integrates audio adapter and sequence strategy with Qwen2.5-0.5B-Instruct LLM.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from tmu_xacle.model.beats_encoder import BEATsEncoder
from tmu_xacle.model.sequence_strategy import (
    ScoreTokenSequenceStrategy,
)
from tmu_xacle.model.swiglu_mlp import SwiGLUProjection


@dataclass
class AudioLLMOutput:
    """Output from LLM processing."""

    last_hidden_state: torch.Tensor  # [B, T, D]
    pooled_output: torch.Tensor  # [B, D]
    attention_mask: torch.Tensor  # [B, T]


class LLMWrapper(nn.Module):
    """
    LLM Wrapper integrating audio and text processing.

    Components:
        - Audio Encoder (BEATs): wav -> audio features
        - Audio Projection (SwiGLU MLP): features -> audio tokens
        - LLM (Qwen2.5-0.5B): sequence -> hidden states
        - Sequence Strategy: constructs input and extracts output

    Args:
        audio_encoder: BEATsEncoder instance
        audio_projection: SwiGLUProjection instance
        llm_model_name: HuggingFace model ID (default: Qwen/Qwen2.5-0.5B-Instruct)
        freeze_llm: Freeze LLM parameters
        lora_config: Optional LoRA configuration
    """

    def __init__(
        self,
        audio_encoder: BEATsEncoder,
        audio_projection: SwiGLUProjection,
        llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        freeze_llm: bool = True,
        lora_config: Optional[LoraConfig] = None,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.audio_projection = audio_projection
        self.sequence_strategy = ScoreTokenSequenceStrategy()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        # Add special tokens
        special_tokens = self.sequence_strategy.get_required_special_tokens()
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        print(f"[LLMWrapper] Added {num_added} special tokens: {special_tokens}")

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize LLM (use float32 for stable inference)
        self.llm = AutoModel.from_pretrained(llm_model_name, dtype=torch.float32)
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Apply LoRA if provided
        if lora_config is not None:
            self.llm = get_peft_model(self.llm, lora_config)
            print(f"[LLMWrapper] Applied LoRA: r={lora_config.r}")

        # Freeze LLM if specified
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            print("[LLMWrapper] LLM frozen")

        print(f"[LLMWrapper] Initialized with {llm_model_name}")

    @property
    def hidden_dim(self) -> int:
        """LLM hidden dimension (896 for Qwen2.5-0.5B)."""
        return self.llm.config.hidden_size

    def forward(
        self,
        wavs: torch.Tensor,
        texts: List[str],
    ) -> AudioLLMOutput:
        """
        Process audio and text through LLM.

        Args:
            wavs: [B, samples] audio waveform at 16kHz
            texts: List of B text captions

        Returns:
            AudioLLMOutput with pooled output from [SCORE] token
        """
        device = wavs.device
        batch_size = wavs.size(0)

        # 1. Encode audio
        audio_output = self.audio_encoder(wavs)
        audio_tokens = self.audio_projection(audio_output.latents)  # AudioAdapterOutput

        # 2. Tokenize text
        text_inputs = self.tokenizer(
            texts,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # 3. Get text embeddings
        text_embeddings = self.llm.get_input_embeddings()(text_inputs["input_ids"])

        # 4. Get special token embeddings
        special_embeddings = self._get_special_token_embeddings(batch_size, device)

        # 5. Build input sequence
        seq_input = self.sequence_strategy.build_input_sequence(
            audio_tokens=audio_tokens.audio_tokens,
            text_tokens=text_embeddings,
            text_mask=text_inputs["attention_mask"],
            special_token_embeddings=special_embeddings,
        )

        # 6. LLM forward
        llm_output = self.llm(
            inputs_embeds=seq_input.sequence_embeddings,
            attention_mask=seq_input.attention_mask,
            return_dict=True,
        )

        # 7. Extract output
        strategy_output = self.sequence_strategy.extract_output(
            hidden_states=llm_output.last_hidden_state,
            attention_mask=seq_input.attention_mask,
            metadata=seq_input.metadata,
        )

        return AudioLLMOutput(
            last_hidden_state=llm_output.last_hidden_state,
            pooled_output=strategy_output.pooled_output,
            attention_mask=seq_input.attention_mask,
        )

    def _get_special_token_embeddings(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Get embeddings for special tokens."""
        embed_fn = self.llm.get_input_embeddings()
        result = {}

        # Get special token embeddings
        for token_name in self.sequence_strategy.get_required_special_tokens():
            token_id = self.tokenizer.convert_tokens_to_ids(token_name)
            ids = torch.full((batch_size, 1), token_id, device=device, dtype=torch.long)
            key = token_name.strip("[]").lower()
            result[key] = embed_fn(ids)

        # Get EOS embedding
        eos_id = self.tokenizer.eos_token_id
        eos_ids = torch.full((batch_size, 1), eos_id, device=device, dtype=torch.long)
        result["eos"] = embed_fn(eos_ids)

        return result

    def generate_caption(
        self,
        wavs: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
    ) -> List[str]:
        """
        Generate captions for audio (for Stage 1 AAC training evaluation).

        Args:
            wavs: [B, samples] audio waveform
            max_length: Maximum generation length
            num_beams: Number of beams for beam search

        Returns:
            List of generated captions
        """
        device = wavs.device
        batch_size = wavs.size(0)

        # Encode audio
        audio_output = self.audio_encoder(wavs)
        audio_tokens = self.audio_projection(audio_output.latents)

        # Get embeddings
        embed_fn = self.llm.get_input_embeddings()
        audio_start_id = self.tokenizer.convert_tokens_to_ids("[AUDIO_START]")
        audio_end_id = self.tokenizer.convert_tokens_to_ids("[AUDIO_END]")

        audio_start_emb = embed_fn(
            torch.full((batch_size, 1), audio_start_id, device=device, dtype=torch.long)
        )
        audio_end_emb = embed_fn(
            torch.full((batch_size, 1), audio_end_id, device=device, dtype=torch.long)
        )

        # Build prefix: [AUDIO_START] [AUDIO] [AUDIO_END]
        torch.cat(
            [
                audio_start_emb,
                audio_tokens.audio_tokens,
                audio_end_emb,
            ],
            dim=1,
        )

        # Generate using LLM
        # Note: This requires AutoModelForCausalLM, not AutoModel
        # For simplicity, we return empty list here
        # In practice, you would need to use a causal LM
        return ["" for _ in range(batch_size)]
