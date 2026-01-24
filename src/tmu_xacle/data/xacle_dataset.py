"""
Dataset for Audio-Text Alignment Score Prediction

Dataset structure:
- CSV columns: wav_file_name, text, average_score
- Audio files: 10-second clips at 16kHz
- Scores: [0, 10] range
"""

from pathlib import Path
from typing import List

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


def normalize_score(score: float) -> float:
    """Normalize score from [0, 10] to [-1, 1]."""
    return (score / 5.0) - 1.0


def denormalize_score(score: float) -> float:
    """Denormalize score from [-1, 1] to [0, 10]."""
    return (score + 1.0) * 5.0


class XACLEDataset(Dataset):
    """
    XACLE Dataset for pointwise alignment score prediction.

    Args:
        csv_path: Path to CSV file with columns (wav_file_name, text, average_score)
        audio_dir: Directory containing audio files
        target_sr: Target sample rate (16000)
        target_length: Target audio length in samples (160000 for 10 seconds)
        normalize: Normalize scores to [-1, 1] range
    """

    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        target_sr: int = 16000,
        target_length: int = 160000,  # 10 seconds at 16kHz
        normalize: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = Path(audio_dir)
        self.target_sr = target_sr
        self.target_length = target_length
        self.normalize = normalize

        print(f"[XACLEDataset] Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            dict with:
                - wav: [samples] audio waveform
                - text: caption string
                - score: normalized score [-1, 1]
                - raw_score: original score [0, 10]
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

        wav = wav.squeeze(0)  # [samples]

        # Pad or truncate to target length
        if wav.size(0) < self.target_length:
            wav = torch.nn.functional.pad(wav, (0, self.target_length - wav.size(0)))
        else:
            wav = wav[: self.target_length]

        # Get score
        raw_score = float(row["average_score"])
        score = normalize_score(raw_score) if self.normalize else raw_score

        return {
            "wav": wav,
            "text": str(row["text"]),
            "score": score,
            "raw_score": raw_score,
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dict with:
            - wavs: [B, samples] audio waveforms
            - texts: List[str] of captions
            - scores: [B] normalized scores
            - raw_scores: [B] original scores
    """
    wavs = torch.stack([item["wav"] for item in batch])
    texts = [item["text"] for item in batch]
    scores = torch.tensor([item["score"] for item in batch], dtype=torch.float32)
    raw_scores = torch.tensor([item["raw_score"] for item in batch], dtype=torch.float32)

    return {
        "wavs": wavs,
        "texts": texts,
        "scores": scores,
        "raw_scores": raw_scores,
    }
