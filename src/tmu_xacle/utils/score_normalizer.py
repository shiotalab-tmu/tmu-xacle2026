"""
Score Normalization Utilities

XACLE scores range from 0 to 10.
For training, we normalize to [-1, 1] range.
"""


def normalize_score(score: float) -> float:
    """
    Normalize score from [0, 10] to [-1, 1].

    Args:
        score: Score in [0, 10] range

    Returns:
        Normalized score in [-1, 1] range
    """
    return (score / 5.0) - 1.0


def denormalize_score(score: float) -> float:
    """
    Denormalize score from [-1, 1] to [0, 10].

    Args:
        score: Normalized score in [-1, 1] range

    Returns:
        Score in [0, 10] range
    """
    return (score + 1.0) * 5.0


def clamp_score(score: float, min_val: float = 0.0, max_val: float = 10.0) -> float:
    """
    Clamp score to valid range.

    Args:
        score: Score value
        min_val: Minimum value (default: 0.0)
        max_val: Maximum value (default: 10.0)

    Returns:
        Clamped score
    """
    return max(min_val, min(max_val, score))
