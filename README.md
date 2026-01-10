<div align="center">

# TMU XACLE System

**Training Large Audio Language Models with CLAP Pseudo-Labels**

[![Paper](https://img.shields.io/badge/ICASSP%202026-Paper-blue)](https://arxiv.org/abs/XXXX.XXXXX)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/Atotti/xacle-tmu-2026)
[![Python](https://img.shields.io/badge/Python-3.11+-yellow.svg)](https://python.org)

<p align="center">
  <img src="https://img.shields.io/badge/XACLE%20Challenge-3rd%20Place-gold?style=for-the-badge" alt="3rd Place">
</p>

*Official implementation of our ICASSP 2026 paper*

[Installation](#installation) | [Quick Start](#quick-start) | [Training](#training-pipeline) | [Models](#pre-trained-models) | [Citation](#citation)

</div>

---

## TODO

- [x] Release training code
- [x] Release inference code
- [x] Release pre-trained models on Hugging Face Hub
- [ ] Paper camera-ready

---

## Overview

We present a **Large Audio Language Model (LALM)** system for predicting semantic alignment between audio and text pairs. Our approach leverages CLAP pseudo-labels for effective pretraining, achieving significant improvements over the baseline.

### Key Results

| Configuration | Val SRCC | Test SRCC |
|:--------------|:--------:|:---------:|
| Official Baseline | 0.384 | 0.334 |
| Our System | 0.674 | 0.625 |
| **Ensemble (Final)** | **0.678** | **0.632** |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/shiotalab-tmu/tmu-xacle2026.git
cd tmu-xacle2026

# Install dependencies
uv sync
```


### Download BEATs Checkpoint

```bash
# Download from Microsoft's official repository
wget https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned.pt \
     -P checkpoints/
```

---

## Quick Start

### Python API

```python
from tmu_xacle.model.xacle_model import XACLEModel

# Load pre-trained model from Hugging Face
model = XACLEModel.from_pretrained(
    "Atotti/xacle-tmu-2026",
    beats_checkpoint="checkpoints/BEATs_iter3_plus_AS2M.pt",
    device="cuda",
)

# Predict alignment score (0-10 scale)
score = model.predict("audio.wav", "A dog barking in the park")
print(f"Alignment Score: {score:.2f}")
```

### Command Line

```bash
# Generate predictions for test set
uv run python scripts/inference.py \
    --checkpoint checkpoints/stage3.pt \
    --beats-checkpoint checkpoints/BEATs_iter3_plus_AS2M_finetuned.pt \
    --csv data/xacle/test.csv \
    --audio-dir data/xacle/wav \
    --output submission.csv

# Evaluate on dev set (with metrics)
uv run python scripts/inference.py \
    --checkpoint checkpoints/stage3.pt \
    --beats-checkpoint checkpoints/BEATs_iter3_plus_AS2M_finetuned.pt \
    --csv data/xacle/dev.csv \
    --audio-dir data/xacle/wav \
    --mode dev
```

---

## Training Pipeline

Our training consists of three stages:

<table>
<tr>
<th>Stage</th>
<th>Task</th>
<th>Data</th>
<th>Epochs</th>
<th>LR</th>
</tr>
<tr>
<td><b>1</b></td>
<td>AAC Pretraining</td>
<td>AudioCaps + VGGSound (273K)</td>
<td>3</td>
<td>1e-5</td>
</tr>
<tr>
<td><b>2</b></td>
<td>CLAP Pseudo-Label</td>
<td>+ Negative Sampling (~1M)</td>
<td>20</td>
<td>1e-5</td>
</tr>
<tr>
<td><b>3</b></td>
<td>XACLE Fine-tuning</td>
<td>XACLE Train (7.5K)</td>
<td>150</td>
<td>6.2e-6</td>
</tr>
</table>

### Stage 2: CLAP Pseudo-Label Pretraining

```bash
uv run python scripts/train.py \
    --stage 2 \
    --train-csv data/clap_pretrain.csv \
    --audio-dir data/audio \
    --beats-checkpoint checkpoints/BEATs_iter3_plus_AS2M_finetuned.pt \
    --epochs 20 \
    --lr 1e-5 \
    --batch-size 16
```

### Stage 3: XACLE Fine-tuning

```bash
uv run python scripts/train.py \
    --stage 3 \
    --train-csv data/xacle/train.csv \
    --val-csv data/xacle/dev.csv \
    --audio-dir data/xacle/wav \
    --beats-checkpoint checkpoints/BEATs_iter3_plus_AS2M_finetuned.pt \
    --checkpoint checkpoints/stage2.pt \
    --epochs 150 \
    --lr 6.2e-6 \
    --freqm 15 --timem 30
```

---

## Pre-trained Models

| Model | Description | Val SRCC | Link |
|:------|:------------|:--------:|:----:|
| **Stage 3** | XACLE fine-tuned | **0.674** | [🤗Atotti/xacle-tmu-2026](https://huggingface.co/Atotti/xacle-tmu-2026) |

```python
from tmu_xacle.model.xacle_model import XACLEModel

model = XACLEModel.from_pretrained(
    "Atotti/xacle-tmu-2026",
    beats_checkpoint="checkpoints/BEATs_iter3_plus_AS2M.pt",
    device="cuda",
)
```

---

## Citation

```bibtex

```
wip
---

## Acknowledgments

wip

---

<div align="center">

**[Tokyo Metropolitan University](https://www.tmu.ac.jp/) - Shiota Laboratory**

</div>
