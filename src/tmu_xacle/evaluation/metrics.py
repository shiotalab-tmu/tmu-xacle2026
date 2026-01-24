"""
Evaluation Metrics

Primary metric: SRCC (Spearman Rank Correlation Coefficient)
Secondary metrics: LCC (Linear Correlation Coefficient), MSE, MAE
"""

from typing import Dict

import numpy as np
from scipy.stats import pearsonr, spearmanr


def srcc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Spearman Rank Correlation Coefficient.

    This is the primary evaluation metric for XACLE.
    It measures how well the rank order of predictions matches the targets.

    Args:
        predictions: Predicted scores
        targets: Ground truth scores

    Returns:
        SRCC value in [-1, 1]
    """
    correlation, _ = spearmanr(predictions, targets)
    return float(correlation)


def lcc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Linear Correlation Coefficient (Pearson).

    Measures the linear relationship between predictions and targets.

    Args:
        predictions: Predicted scores
        targets: Ground truth scores

    Returns:
        LCC value in [-1, 1]
    """
    correlation, _ = pearsonr(predictions, targets)
    return float(correlation)


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Mean Squared Error.

    Args:
        predictions: Predicted scores
        targets: Ground truth scores

    Returns:
        MSE value
    """
    return float(np.mean((predictions - targets) ** 2))


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        predictions: Predicted scores
        targets: Ground truth scores

    Returns:
        MAE value
    """
    return float(np.mean(np.abs(predictions - targets)))


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        predictions: Predicted scores
        targets: Ground truth scores

    Returns:
        Dict with SRCC, LCC, MSE, MAE
    """
    return {
        "srcc": srcc(predictions, targets),
        "lcc": lcc(predictions, targets),
        "mse": mse(predictions, targets),
        "mae": mae(predictions, targets),
    }
