"""
BEATs Audio Encoder with SpecAugment Support

BEATs: Audio Pre-Training with Acoustic Tokenizers
https://arxiv.org/abs/2212.09058
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchaudio.transforms as T
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.beats import BEATs as SpeechBrainBEATs


@dataclass
class AudioEncoderOutput:
    """Output from audio encoder."""

    latents: torch.Tensor  # [B, T, D] sequence features
    embeddings: torch.Tensor  # [B, D] pooled features
    latent_dim: int
    embedding_dim: int


class SpecAugment(nn.Module):
    """
    SpecAugment: Time and Frequency Masking for Spectrograms

    Reference:
        Park et al. "SpecAugment: A Simple Data Augmentation Method for ASR"
        https://arxiv.org/abs/1904.08779
    """

    def __init__(self, freqm: int = 0, timem: int = 0):
        super().__init__()
        self.freqm = freqm
        self.timem = timem
        self.freqmask = T.FrequencyMasking(freqm) if freqm > 0 else None
        self.timemask = T.TimeMasking(timem) if timem > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram (B, F, T)."""
        if self.freqmask is not None:
            x = self.freqmask(x)
        if self.timemask is not None:
            x = self.timemask(x)
        return x

    @staticmethod
    def is_enabled(freqm: int, timem: int) -> bool:
        return freqm > 0 or timem > 0


class _BEATsWithSpecAugment(SpeechBrainBEATs):
    """Internal BEATs class with SpecAugment integration."""

    def __init__(self, *args, spec_augment: Optional[SpecAugment] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec_augment = spec_augment

    def extract_features(
        self,
        wav: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> torch.Tensor:
        """Extract features with optional SpecAugment."""
        # Preprocess
        fbank = self.preprocess(wav, fbank_mean, fbank_std)

        # Apply SpecAugment during training
        if self.spec_augment is not None and self.training:
            fbank = fbank.transpose(1, 2)
            fbank = self.spec_augment(fbank)
            fbank = fbank.transpose(1, 2)

        # Padding mask
        if wav_lens is not None:
            max_len = wav.size(-1)
            padding_mask = ~length_to_mask(wav_lens * max_len, max_len, device=wav.device).bool()
        else:
            padding_mask = None

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1).transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        x, layer_results = self.encoder(
            features,
            padding_mask=padding_mask,
            output_all_hiddens=self.output_all_hiddens,
        )

        if self.predictor is not None:
            x_d = self.predictor_dropout(x)
            logits = self.predictor(x_d)

            if padding_mask is not None and padding_mask.any():
                logits[padding_mask] = 0
                logits = logits.sum(dim=1)
                logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
            else:
                logits = logits.mean(dim=1)

            lprobs = torch.sigmoid(logits)

            if self.output_all_hiddens:
                x = torch.stack(layer_results, dim=0)
            return x, lprobs, padding_mask

        if self.output_all_hiddens:
            x = torch.stack(layer_results, dim=0)

        return (x,)


class BEATsEncoder(nn.Module):
    """
    BEATs Audio Encoder with SpecAugment support.

    Args:
        checkpoint_path: Path to BEATs checkpoint (BEATs_iter3_plus_AS2M_finetuned.pt)
        freeze: Freeze encoder weights
        feature_dim: Feature dimension (768 for base model)
        freqm: Frequency mask size for SpecAugment (0 to disable)
        timem: Time mask size for SpecAugment (0 to disable)
    """

    def __init__(
        self,
        checkpoint_path: str,
        freeze: bool = True,
        feature_dim: int = 768,
        freqm: int = 0,
        timem: int = 0,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self._feature_dim = feature_dim
        self._freeze = freeze
        self._device = None

        # Initialize SpecAugment
        spec_augment = None
        if SpecAugment.is_enabled(freqm, timem):
            spec_augment = SpecAugment(freqm=freqm, timem=timem)
            print(f"[BEATsEncoder] SpecAugment enabled: freqm={freqm}, timem={timem}")

        # Initialize BEATs
        self.model = _BEATsWithSpecAugment(
            ckp_path=checkpoint_path,
            freeze=freeze,
            output_all_hiddens=False,
            spec_augment=spec_augment,
        )

        # Get actual feature dimension from config
        if hasattr(self.model, "cfg"):
            self._feature_dim = self.model.cfg.encoder_embed_dim

        # Freeze if specified
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        print(f"[BEATsEncoder] Loaded from {checkpoint_path}")
        print(f"[BEATsEncoder] Feature dim: {self._feature_dim}, Frozen: {freeze}")

    @property
    def latent_dim(self) -> int:
        return self._feature_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def _apply(self, fn):
        super()._apply(fn)
        self._device = next(self.parameters()).device
        return self

    def forward(self, wavs: torch.Tensor) -> AudioEncoderOutput:
        """
        Encode audio waveforms.

        Args:
            wavs: [B, T] audio waveform at 16kHz

        Returns:
            AudioEncoderOutput with latents [B, T_audio, D] and embeddings [B, D]
        """
        wavs = wavs.to(self.device)

        if wavs.dim() == 3:
            wavs = wavs.squeeze(1)

        batch_size = wavs.size(0)
        length = torch.ones(batch_size, device=self.device)

        if self._freeze:
            with torch.no_grad():
                result = self.model.extract_features(wavs, length)
        else:
            result = self.model.extract_features(wavs, length)

        latents = result[0] if isinstance(result, tuple) else result

        # Mean pooling for embeddings
        embeddings = latents.mean(dim=1)

        return AudioEncoderOutput(
            latents=latents,
            embeddings=embeddings,
            latent_dim=self._feature_dim,
            embedding_dim=self._feature_dim,
        )

    def enable_augmentation(self, freqm: int, timem: int):
        """Enable SpecAugment with given parameters."""
        if SpecAugment.is_enabled(freqm, timem):
            self.model.spec_augment = SpecAugment(freqm=freqm, timem=timem)
            print(f"[BEATsEncoder] SpecAugment enabled: freqm={freqm}, timem={timem}")

    def disable_augmentation(self):
        """Disable SpecAugment."""
        self.model.spec_augment = None
