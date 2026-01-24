"""
XACLE Trainer using PyTorch Lightning

Supports:
- Stage 2: CLAP pseudo-label pretraining with ListNet
- Stage 3: XACLE fine-tuning with ListNet and SpecAugment
"""

from typing import Any, Dict

import pytorch_lightning as pl
import torch
from scipy.stats import spearmanr

from tmu_xacle.model.xacle_model import XACLEModel
from tmu_xacle.training.listnet_loss import ListNetLoss


class XACLETrainer(pl.LightningModule):
    """
    PyTorch Lightning module for XACLE training.

    Supports both Stage 2 (CLAP pretrain) and Stage 3 (XACLE fine-tune).

    Args:
        model: XACLEModel instance
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        temperature: Temperature for ListNet loss
    """

    def __init__(
        self,
        model: XACLEModel,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss function
        self.loss_fn = ListNetLoss(temperature=temperature)

        # For validation metrics
        self.val_predictions = []
        self.val_targets = []

        self.save_hyperparameters(ignore=["model"])

    def forward(self, wavs, texts):
        return self.model(wavs, texts)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        wavs = batch["wavs"]
        texts = batch["texts"]
        targets = batch["scores"]

        # Forward pass
        output = self.model(wavs, texts)
        predictions = output.scores

        # Compute loss
        loss_output = self.loss_fn(predictions, targets)
        loss = loss_output.loss

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step."""
        wavs = batch["wavs"]
        texts = batch["texts"]
        targets = batch["scores"]

        # Forward pass
        output = self.model(wavs, texts)
        predictions = output.scores

        # Compute loss
        loss_output = self.loss_fn(predictions, targets)
        loss = loss_output.loss

        # Collect predictions for SRCC computation
        self.val_predictions.extend(predictions.cpu().tolist())
        self.val_targets.extend(targets.cpu().tolist())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """Compute SRCC at end of validation epoch."""
        if len(self.val_predictions) > 1:
            srcc, _ = spearmanr(self.val_predictions, self.val_targets)
            self.log("val/srcc", srcc, prog_bar=True)

        # Reset
        self.val_predictions = []
        self.val_targets = []

    def configure_optimizers(self):
        """Configure optimizer."""
        # Only train unfrozen parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class SFTTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for Stage 1 SFT (Supervised Fine-Tuning).

    Uses language modeling loss for audio captioning pretraining.

    Note: This requires AutoModelForCausalLM instead of AutoModel.
    For simplicity, this is a placeholder that would need to be
    implemented with proper causal LM support.
    """

    def __init__(
        self,
        model: XACLEModel,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["model"])

    def forward(self, wavs, texts):
        return self.model(wavs, texts)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step for SFT.

        Note: Full implementation would require:
        1. AutoModelForCausalLM instead of AutoModel
        2. Proper teacher forcing with captions
        3. Cross-entropy loss over vocabulary
        """
        # Placeholder - actual SFT would compute cross-entropy loss
        wavs = batch["wavs"]
        texts = batch["texts"]

        # For now, just return a dummy loss
        # Real implementation would compute LM loss
        output = self.model(wavs, texts)
        loss = output.scores.mean() * 0  # Dummy zero loss

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        return optimizer
