"""
CLAP Pretrain Dataset for Stage 2 Training

Dataset with CLAP pseudo-labels for alignment score prediction.
Includes negative sampling for data augmentation.
"""

from pathlib import Path
from typing import List

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class CLAPPretrainDataset(Dataset):
    """
    CLAP Pretrain Dataset with pseudo-labels.

    CSV columns:
        - wav_file_name: audio file name
        - text: caption text
        - clap_score: CLAP similarity score (pseudo-label)
        - is_negative: 0 for positive, 1 for negative sample (optional)

    Args:
        csv_path: Path to CSV file with CLAP scores
        audio_dir: Directory containing audio files
        target_sr: Target sample rate (16000)
        target_length: Target audio length (160000 for 10 seconds)
        include_negatives: Include negative samples in dataset
    """

    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        target_sr: int = 16000,
        target_length: int = 160000,
        include_negatives: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = Path(audio_dir)
        self.target_sr = target_sr
        self.target_length = target_length

        # Filter out negatives if not including
        if not include_negatives and "is_negative" in self.df.columns:
            self.df = self.df[self.df["is_negative"] == 0].reset_index(drop=True)

        print(f"[CLAPPretrainDataset] Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            dict with:
                - wav: [samples] audio waveform
                - text: caption string
                - score: CLAP score (normalized to [-1, 1])
        """
        row = self.df.iloc[idx]

        # Load audio
        wav_path = self.audio_dir / row["wav_file_name"]
        wav, sr = torchaudio.load(wav_path)

        # Resample if needed
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        # Convert stereo to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = wav.squeeze(0)

        # Pad or truncate
        if wav.size(0) < self.target_length:
            wav = torch.nn.functional.pad(wav, (0, self.target_length - wav.size(0)))
        else:
            wav = wav[: self.target_length]

        # CLAP score is typically in [-1, 1] range already
        clap_score = float(row["clap_score"])

        return {
            "wav": wav,
            "text": str(row["text"]),
            "score": clap_score,
        }


def collate_fn(batch: List[dict]) -> dict:
    """Collate function for DataLoader."""
    wavs = torch.stack([item["wav"] for item in batch])
    texts = [item["text"] for item in batch]
    scores = torch.tensor([item["score"] for item in batch], dtype=torch.float32)

    return {
        "wavs": wavs,
        "texts": texts,
        "scores": scores,
    }
