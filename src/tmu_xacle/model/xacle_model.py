"""
XACLE Model - Main Model Class

TMU System for XACLE Challenge
Audio-Text Alignment Score Prediction
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from tmu_xacle.model.beats_encoder import BEATsEncoder
from tmu_xacle.model.swiglu_mlp import SwiGLUProjection
from tmu_xacle.model.llm_wrapper import LLMWrapper
from tmu_xacle.model.score_head import ScoreHead, LinearScoreHead


@dataclass
class XACLEOutput:
    """Output from XACLE model."""

    scores: torch.Tensor  # [B] alignment scores (normalized to [-1, 1])
    logits: torch.Tensor  # [B, logits_dim] raw logits before final projection


class XACLEModel(nn.Module):
    """
    XACLE Model for Audio-Text Alignment Score Prediction.

    Architecture:
        Audio Waveform
              |
        BEATs Encoder (frozen, 768-dim, 500 tokens at 50Hz)
              |
        SwiGLU MLP Projection (768 -> 896, 500 -> 100 tokens)
              |
        Qwen2.5-0.5B LLM (frozen, 896-dim)
              |
        [SCORE] Token Hidden State
              |
        Score Head MLP (896 -> 512 -> 128)
              |
        Linear (128 -> 1)
              |
        Alignment Score [-1, 1]

    Total parameters: ~594M
        - BEATs: 90M (frozen)
        - SwiGLU MLP: ~10M (trainable)
        - Qwen2.5-0.5B: ~494M (frozen)
        - Score Head: ~0.3M (trainable)

    Args:
        beats_checkpoint: Path to BEATs checkpoint
        llm_model_name: HuggingFace model ID for LLM
        freeze_audio_encoder: Freeze BEATs parameters
        freeze_llm: Freeze LLM parameters
        freqm: Frequency mask size for SpecAugment (0 to disable)
        timem: Time mask size for SpecAugment (0 to disable)
    """

    def __init__(
        self,
        beats_checkpoint: str,
        llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        freeze_audio_encoder: bool = True,
        freeze_llm: bool = True,
        freqm: int = 0,
        timem: int = 0,
        num_audio_tokens: int = 100,
    ):
        super().__init__()

        # Audio encoder
        audio_encoder = BEATsEncoder(
            checkpoint_path=beats_checkpoint,
            freeze=freeze_audio_encoder,
            feature_dim=768,
            freqm=freqm,
            timem=timem,
        )

        # Audio projection (SwiGLU MLP with gated residual)
        audio_projection = SwiGLUProjection(
            audio_dim=768,  # BEATs output
            llm_hidden_dim=896,  # Qwen2.5-0.5B hidden dim
            num_tokens=num_audio_tokens,
            intermediate_dim=3584,  # 4 * 896 (matches xacle)
            dropout=0.1,
            use_residual_gate=True,
        )

        # LLM wrapper (owns audio_encoder and audio_projection)
        self.llm_wrapper = LLMWrapper(
            audio_encoder=audio_encoder,
            audio_projection=audio_projection,
            llm_model_name=llm_model_name,
            freeze_llm=freeze_llm,
        )

        # Score head
        self.score_head = ScoreHead(
            input_dim=896,  # LLM hidden dim
            hidden_dim=512,
            output_dim=128,
            activation="ReLU",
            dropout=0.1,
        )

        # Final linear layer
        self.final_linear = LinearScoreHead(input_dim=128)

        print(f"[XACLEModel] Initialized with:")
        print(f"  Audio Encoder: BEATs (frozen={freeze_audio_encoder})")
        print(f"  LLM: {llm_model_name} (frozen={freeze_llm})")
        print(f"  SpecAugment: freqm={freqm}, timem={timem}")

    def forward(
        self,
        wavs: torch.Tensor,
        texts: List[str],
    ) -> XACLEOutput:
        """
        Forward pass: Audio + Text -> Alignment Score

        Args:
            wavs: [B, samples] audio waveform at 16kHz (10 seconds = 160000 samples)
            texts: List of B text captions

        Returns:
            XACLEOutput with scores and logits
        """
        # 1. LLM processing
        llm_output = self.llm_wrapper(wavs, texts)

        # 2. Score head projection
        logits = self.score_head(llm_output.pooled_output)  # [B, 128]

        # 3. Final score prediction
        scores = self.final_linear(logits)  # [B]

        return XACLEOutput(
            scores=scores,
            logits=logits,
        )

    def predict(
        self,
        audio_path: str,
        text: str,
        device: str = "cuda",
    ) -> float:
        """
        Predict alignment score for a single audio-text pair.

        Args:
            audio_path: Path to audio file
            text: Text caption
            device: Device to run on

        Returns:
            Alignment score (denormalized to [0, 10] scale)
        """
        import torchaudio

        # Load audio
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        # Handle stereo to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Ensure 10 seconds
        target_length = 160000
        if wav.size(1) < target_length:
            wav = torch.nn.functional.pad(wav, (0, target_length - wav.size(1)))
        else:
            wav = wav[:, :target_length]

        wav = wav.squeeze(0).unsqueeze(0).to(device)  # [1, samples]

        # Ensure model is on correct device
        self.to(device)

        # Predict
        self.eval()
        with torch.no_grad():
            output = self.forward(wav, [text])
            score = output.scores.item()

        # Denormalize from [-1, 1] to [0, 10]
        score = (score + 1) * 5

        return score

    def enable_augmentation(self, freqm: int = 15, timem: int = 30):
        """Enable SpecAugment with given parameters."""
        self.llm_wrapper.audio_encoder.enable_augmentation(freqm, timem)

    def disable_augmentation(self):
        """Disable SpecAugment."""
        self.llm_wrapper.audio_encoder.disable_augmentation()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        beats_checkpoint: str,
        llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        freeze_llm: bool = True,
    ) -> "XACLEModel":
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to saved checkpoint
            beats_checkpoint: Path to BEATs checkpoint
            llm_model_name: HuggingFace model ID
            device: Device to load model on
            freeze_llm: Freeze LLM parameters after loading

        Returns:
            Loaded XACLEModel
        """
        model = cls(
            beats_checkpoint=beats_checkpoint,
            llm_model_name=llm_model_name,
            freeze_audio_encoder=True,
            freeze_llm=False,  # Load first, then freeze if needed
        )

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Map xacle checkpoint keys to our model keys
        key_mapping = {
            # Audio adapter -> audio_projection
            "model.model.llm_wrapper.audio_adapter.mlp.": "llm_wrapper.audio_projection.mlp.",
            "model.model.llm_wrapper.audio_adapter.residual_proj.": "llm_wrapper.audio_projection.residual_proj.",
            "model.model.llm_wrapper.audio_adapter.residual_gate.": "llm_wrapper.audio_projection.residual_gate.",
            # Audio encoder
            "model.model.llm_wrapper.audio_adapter.audio_encoder.": "llm_wrapper.audio_encoder.",
            # LLM
            "model.model.llm_wrapper.llm.": "llm_wrapper.llm.",
            # Score head
            "model.model.projection.": "score_head.",
            # Final linear (xacle uses nn.Linear directly, we use LinearScoreHead.linear)
            "model.projection.weight": "final_linear.linear.weight",
            "model.projection.bias": "final_linear.linear.bias",
        }

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            # Check for exact key match first (for final linear)
            if k in key_mapping:
                new_key = key_mapping[k]
            else:
                # Then check for prefix match
                for old_prefix, new_prefix in key_mapping.items():
                    if k.startswith(old_prefix) and not old_prefix.endswith("weight") and not old_prefix.endswith("bias"):
                        new_key = new_prefix + k[len(old_prefix):]
                        break
            new_state_dict[new_key] = v

        # Load state dict
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        # Freeze LLM if requested
        if freeze_llm:
            for param in model.llm_wrapper.llm.parameters():
                param.requires_grad = False

        model.to(device)
        model.eval()

        print(f"[XACLEModel] Loaded from {checkpoint_path}")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

        return model

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        beats_checkpoint: Optional[str] = None,
        device: str = "cuda",
    ) -> "XACLEModel":
        """
        Load model from Hugging Face Hub.

        Args:
            repo_id: Hugging Face repository ID (e.g., 'Atotti/xacle-tmu-2026')
            beats_checkpoint: Path to BEATs checkpoint. If None, downloads automatically.
            device: Device to load model on

        Returns:
            Loaded XACLEModel
        """
        import json
        from huggingface_hub import hf_hub_download

        print(f"[XACLEModel] Loading from {repo_id}...")

        # Download config and model from HF
        config_path = hf_hub_download(repo_id, "config.json")
        model_path = hf_hub_download(repo_id, "model.pt")

        # Load config
        with open(config_path) as f:
            config = json.load(f)

        # Get BEATs checkpoint
        if beats_checkpoint is None:
            raise ValueError(
                "beats_checkpoint is required. Download from:\n"
                "  https://github.com/microsoft/unilm/tree/master/beats\n"
                "  wget https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt"
            )

        # Create model
        model = cls(
            beats_checkpoint=beats_checkpoint,
            llm_model_name=config.get("llm_model_name", "Qwen/Qwen2.5-0.5B-Instruct"),
            freeze_audio_encoder=True,
            freeze_llm=True,
            num_audio_tokens=config.get("num_audio_tokens", 100),
        )

        # Load weights
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)

        model.to(device)
        model.eval()

        print(f"[XACLEModel] Loaded from {repo_id}")
        print(f"  Val SRCC: {config.get('val_srcc', 'N/A')}")

        return model

    def save_checkpoint(
        self,
        path: str,
        epoch: Optional[int] = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
    ):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch (optional)
            optimizer_state: Optimizer state dict (optional)
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
        }
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"[XACLEModel] Saved checkpoint to {path}")
