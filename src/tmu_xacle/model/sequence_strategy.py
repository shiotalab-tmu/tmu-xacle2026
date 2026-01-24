"""
Score Token Sequence Strategy

Constructs input sequence in the format:
[TEXT] [AUDIO_START] [AUDIO] [AUDIO_END] [SCORE] [EOS]

The [SCORE] token is used to extract the alignment score.
"""

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class SequenceStrategyInput:
    """Input sequence for LLM."""

    sequence_embeddings: torch.Tensor  # [B, T, D]
    attention_mask: torch.Tensor  # [B, T]
    metadata: Dict[str, Any]  # Position indices


@dataclass
class SequenceStrategyOutput:
    """Output from sequence strategy."""

    pooled_output: torch.Tensor  # [B, D]
    auxiliary_outputs: Dict[str, Any]


class ScoreTokenSequenceStrategy:
    """
    Score Token Sequence Strategy.

    Sequence format:
        [TEXT] [AUDIO_START] [AUDIO] [AUDIO_END] [SCORE] [EOS]

    The [SCORE] token's hidden state is used as the output for score prediction.
    This is effective because causal attention allows the [SCORE] token to
    attend to all previous tokens (text + audio).
    """

    SPECIAL_TOKENS = ["[AUDIO_START]", "[AUDIO_END]", "[SCORE]"]

    def __init__(self):
        pass

    def get_required_special_tokens(self):
        """Return special tokens needed for this strategy."""
        return self.SPECIAL_TOKENS

    def requires_eos(self) -> bool:
        """This strategy requires EOS token."""
        return True

    def build_input_sequence(
        self,
        audio_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        special_token_embeddings: Dict[str, torch.Tensor],
    ) -> SequenceStrategyInput:
        """
        Build input sequence: [TEXT] [AUDIO_START] [AUDIO] [AUDIO_END] [SCORE] [EOS]

        Args:
            audio_tokens: [B, N, D] audio token embeddings
            text_tokens: [B, T_text, D] text token embeddings
            text_mask: [B, T_text] text attention mask
            special_token_embeddings: Dict with 'audio_start', 'audio_end', 'score', 'eos'

        Returns:
            SequenceStrategyInput with embeddings, mask, and metadata
        """
        B, T_text = text_tokens.shape[:2]
        N = audio_tokens.size(1)
        device = text_tokens.device

        # Concatenate sequence
        sequence_embeddings = torch.cat(
            [
                text_tokens,  # [B, T_text, D]
                special_token_embeddings["audio_start"],  # [B, 1, D]
                audio_tokens,  # [B, N, D]
                special_token_embeddings["audio_end"],  # [B, 1, D]
                special_token_embeddings["score"],  # [B, 1, D]
                special_token_embeddings["eos"],  # [B, 1, D]
            ],
            dim=1,
        )  # [B, T_text + N + 4, D]

        # Build attention mask
        ones = torch.ones(B, N + 4, device=device, dtype=torch.long)
        attention_mask = torch.cat([text_mask, ones], dim=1)

        # Calculate indices
        # [TEXT: 0 ~ T_text-1]
        # [AUDIO_START: T_text]
        # [AUDIO: T_text+1 ~ T_text+N]
        # [AUDIO_END: T_text+N+1]
        # [SCORE: T_text+N+2]
        # [EOS: T_text+N+3]
        score_idx = T_text + N + 2
        eos_idx = score_idx + 1

        return SequenceStrategyInput(
            sequence_embeddings=sequence_embeddings,
            attention_mask=attention_mask,
            metadata={
                "score_idx": score_idx,
                "eos_idx": eos_idx,
                "T_text": T_text,
                "N_audio": N,
            },
        )

    def extract_output(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> SequenceStrategyOutput:
        """
        Extract [SCORE] token's hidden state as output.

        Args:
            hidden_states: [B, T, D] LLM output
            attention_mask: [B, T]
            metadata: Contains 'score_idx'

        Returns:
            SequenceStrategyOutput with pooled output
        """
        score_idx = metadata["score_idx"]
        pooled = hidden_states[:, score_idx, :]  # [B, D]

        return SequenceStrategyOutput(
            pooled_output=pooled,
            auxiliary_outputs={},
        )
