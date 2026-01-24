"""
Inference Script

Generate alignment score predictions for audio-text pairs.

Usage:
    # Dev evaluation (with metrics)
    python scripts/inference.py \
        --checkpoint checkpoints/stage3.pt \
        --beats-checkpoint checkpoints/BEATs_iter3_plus_AS2M_finetuned.pt \
        --csv data/xacle/dev.csv \
        --audio-dir data/xacle/wav \
        --mode dev \
        --output predictions.csv

    # Test inference (submission)
    python scripts/inference.py \
        --checkpoint checkpoints/stage3.pt \
        --beats-checkpoint checkpoints/BEATs_iter3_plus_AS2M_finetuned.pt \
        --csv data/xacle/test.csv \
        --audio-dir data/xacle/wav \
        --mode test \
        --output submission.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tmu_xacle.data.xacle_dataset import denormalize_score
from tmu_xacle.evaluation.metrics import compute_metrics
from tmu_xacle.model.xacle_model import XACLEModel


def parse_args():
    parser = argparse.ArgumentParser(description="XACLE Inference")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--beats-checkpoint",
        type=str,
        required=True,
        help="Path to BEATs checkpoint",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="LLM model name",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file (wav_file_name, text, [average_score])",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory with audio files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "test"],
        default="test",
        help="dev: compute metrics, test: just predict",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[Inference] Loading model from {args.checkpoint}")

    # Load model
    model = XACLEModel.from_checkpoint(
        checkpoint_path=args.checkpoint,
        beats_checkpoint=args.beats_checkpoint,
        llm_model_name=args.llm_model,
        device=args.device,
    )
    model.eval()

    # Load data
    df = pd.read_csv(args.csv)
    audio_dir = Path(args.audio_dir)

    print(f"[Inference] Processing {len(df)} samples")

    predictions = []
    targets = [] if args.mode == "dev" else None

    import torchaudio

    # Process in batches
    for i in tqdm(range(0, len(df), args.batch_size)):
        batch_df = df.iloc[i : i + args.batch_size]

        wavs = []
        texts = []

        for _, row in batch_df.iterrows():
            # Load audio
            wav_path = audio_dir / row["wav_file_name"]
            wav, sr = torchaudio.load(wav_path)

            # Resample if needed
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)

            # Convert to mono
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)

            wav = wav.squeeze(0)

            # Pad/truncate to 10 seconds
            target_length = 160000
            if wav.size(0) < target_length:
                wav = torch.nn.functional.pad(wav, (0, target_length - wav.size(0)))
            else:
                wav = wav[:target_length]

            wavs.append(wav)
            texts.append(str(row["text"]))

            if args.mode == "dev" and "average_score" in row:
                targets.append(float(row["average_score"]))

        # Stack and move to device
        wavs = torch.stack(wavs).to(args.device)

        # Predict
        with torch.no_grad():
            output = model(wavs, texts)
            batch_preds = output.scores.cpu().numpy()

        # Denormalize predictions
        batch_preds = [denormalize_score(p) for p in batch_preds]
        predictions.extend(batch_preds)

    # Add predictions to dataframe
    df["prediction_score"] = predictions

    # Compute metrics if dev mode
    if args.mode == "dev" and targets:
        predictions_arr = np.array(predictions)
        targets_arr = np.array(targets)

        metrics = compute_metrics(predictions_arr, targets_arr)

        print("\n[Results]")
        print(f"  SRCC: {metrics['srcc']:.4f}")
        print(f"  LCC:  {metrics['lcc']:.4f}")
        print(f"  MSE:  {metrics['mse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")

    # Save predictions
    df.to_csv(args.output, index=False)
    print(f"\n[Inference] Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
