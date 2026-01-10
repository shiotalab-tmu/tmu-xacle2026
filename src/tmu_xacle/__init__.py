"""
TMU XACLE System - Audio-Text Alignment Score Prediction

Official implementation for ICASSP 2026 paper:
"The TMU System for the XACLE Challenge: Training Large Audio Language Models with CLAP Pseudo-Labels"
"""

__version__ = "1.0.0"

from tmu_xacle.model.xacle_model import XACLEModel

__all__ = ["XACLEModel"]
