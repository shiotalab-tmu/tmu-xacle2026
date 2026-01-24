"""
ListNet Loss for Ranking-based Learning

ListNet is a listwise learning-to-rank loss that optimizes the top-1 probability
distribution alignment between predictions and targets.

Reference:
    Cao et al. "Learning to Rank: From Pairwise Approach to Listwise Approach"
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutput:
    """Output from loss function."""

    loss: torch.Tensor
    components: dict


class ListNetLoss(nn.Module):
    """
    ListNet Top-1 Probability Loss.

    Computes KL divergence between the top-1 probability distributions
    of predictions and targets.

    The top-1 probability of item i is:
        P(i) = exp(s_i / temperature) / sum_j(exp(s_j / temperature))

    Loss = KL(P_target || P_pred) = -sum_i(P_target_i * log(P_pred_i))

    Args:
        temperature: Temperature for softmax (default: 1.0)
        reduction: Loss reduction ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> LossOutput:
        """
        Compute ListNet loss.

        Args:
            predictions: [B] predicted scores
            targets: [B] target scores

        Returns:
            LossOutput with loss tensor and components dict
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Compute top-1 probabilities
        target_probs = F.softmax(targets / self.temperature, dim=0)
        pred_log_probs = F.log_softmax(predictions, dim=0)

        # Cross-entropy loss (KL divergence up to a constant)
        loss = -(target_probs * pred_log_probs).sum()

        return LossOutput(
            loss=loss,
            components={"listnet": loss.item()},
        )


class GroupListNetLoss(nn.Module):
    """
    Group-aware ListNet Loss.

    Computes ListNet loss within groups (e.g., same audio with different captions).
    This is useful when samples can be grouped by queries.

    Args:
        temperature: Temperature for softmax
        reduction: Reduction method for group losses
    """

    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> LossOutput:
        """
        Compute group-aware ListNet loss.

        Args:
            predictions: [B] predicted scores
            targets: [B] target scores
            group_ids: [B] group identifiers

        Returns:
            LossOutput with loss tensor and components dict
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        group_ids = group_ids.view(-1)

        unique_groups = torch.unique(group_ids)
        group_losses = []

        for gid in unique_groups:
            mask = group_ids == gid
            if mask.sum() < 2:
                continue

            group_preds = predictions[mask]
            group_targets = targets[mask]

            target_probs = F.softmax(group_targets / self.temperature, dim=0)
            pred_log_probs = F.log_softmax(group_preds, dim=0)

            group_loss = -(target_probs * pred_log_probs).sum()
            group_losses.append(group_loss)

        if not group_losses:
            loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        else:
            stacked = torch.stack(group_losses)
            if self.reduction == "mean":
                loss = stacked.mean()
            elif self.reduction == "sum":
                loss = stacked.sum()
            else:
                loss = stacked

        return LossOutput(
            loss=loss,
            components={"listnet_group": loss.item() if loss.dim() == 0 else loss.mean().item()},
        )
