#!/usr/bin/env python3
"""
XACLE Training Script

Supports:
- Stage 2: CLAP pseudo-label pretraining
- Stage 3: XACLE fine-tuning with SpecAugment

Usage:
    # Stage 2: CLAP pretrain
    python scripts/train.py \
        --stage 2 \
        --train-csv data/clap_pretrain.csv \
        --audio-dir data/audio \
        --beats-checkpoint checkpoints/BEATs_iter3_plus_AS2M_finetuned.pt \
        --epochs 20

    # Stage 3: XACLE fine-tune
    python scripts/train.py \
        --stage 3 \
        --train-csv data/xacle/train.csv \
        --val-csv data/xacle/dev.csv \
        --audio-dir data/xacle/wav \
        --beats-checkpoint checkpoints/BEATs_iter3_plus_AS2M_finetuned.pt \
        --checkpoint checkpoints/stage2.pt \
        --epochs 150 \
        --freqm 15 --timem 30
"""

import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tmu_xacle.model.xacle_model import XACLEModel
from tmu_xacle.data.xacle_dataset import XACLEDataset, collate_fn
from tmu_xacle.data.clap_dataset import CLAPPretrainDataset
from tmu_xacle.data.clap_dataset import collate_fn as clap_collate_fn
from tmu_xacle.training.trainer import XACLETrainer


def parse_args():
    parser = argparse.ArgumentParser(description="XACLE Training")

    # Data
    parser.add_argument("--train-csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--val-csv", type=str, default=None, help="Path to validation CSV")
    parser.add_argument("--audio-dir", type=str, required=True, help="Directory with audio files")

    # Model
    parser.add_argument(
        "--beats-checkpoint",
        type=str,
        required=True,
        help="Path to BEATs checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint (for Stage 3)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="LLM model name",
    )

    # Training
    parser.add_argument("--stage", type=int, choices=[2, 3], required=True, help="Training stage")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=1.0, help="ListNet temperature")

    # SpecAugment (Stage 3)
    parser.add_argument("--freqm", type=int, default=0, help="Frequency mask size")
    parser.add_argument("--timem", type=int, default=0, help="Time mask size")

    # Hardware
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision")

    # Output
    parser.add_argument("--output-dir", type=str, default="runs", help="Output directory")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup experiment name
    if args.exp_name is None:
        args.exp_name = f"stage{args.stage}_lr{args.lr}_bs{args.batch_size}"

    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Train] Stage {args.stage} training")
    print(f"[Train] Output: {output_dir}")

    # Create model
    model = XACLEModel(
        beats_checkpoint=args.beats_checkpoint,
        llm_model_name=args.llm_model,
        freeze_audio_encoder=True,
        freeze_llm=True,
        freqm=args.freqm,
        timem=args.timem,
    )

    # Load checkpoint if resuming
    if args.checkpoint is not None:
        print(f"[Train] Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)

    # Create datasets
    if args.stage == 2:
        train_dataset = CLAPPretrainDataset(
            csv_path=args.train_csv,
            audio_dir=args.audio_dir,
        )
        collate = clap_collate_fn
    else:  # Stage 3
        train_dataset = XACLEDataset(
            csv_path=args.train_csv,
            audio_dir=args.audio_dir,
        )
        collate = collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
        pin_memory=True,
    )

    val_loader = None
    if args.val_csv is not None:
        val_dataset = XACLEDataset(
            csv_path=args.val_csv,
            audio_dir=args.audio_dir,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # Create trainer module
    trainer_module = XACLETrainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
    )

    # Callbacks
    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="model-{epoch:02d}-{val/srcc:.4f}" if val_loader else "model-{epoch:02d}",
            save_top_k=3,
            monitor="val/srcc" if val_loader else "train/loss",
            mode="max" if val_loader else "min",
        ),
    ]

    if val_loader:
        callbacks.append(
            EarlyStopping(
                monitor="val/srcc",
                patience=20,
                mode="max",
            )
        )

    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
    )

    # Train
    trainer.fit(trainer_module, train_loader, val_loader)

    # Save final checkpoint
    final_path = output_dir / "checkpoints" / "final.pt"
    model.save_checkpoint(str(final_path), epoch=args.epochs)

    print(f"[Train] Training complete! Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
