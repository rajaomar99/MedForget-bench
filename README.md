# MedForget-bench

> **An Open Evaluation Harness for Machine Unlearning in Medical Imaging**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*FAST NUCES Lahore — Undergraduate Research Project*

---

## Motivation

Privacy regulations such as GDPR and HIPAA give patients the right to request deletion of their data from AI systems. For a deployed medical imaging classifier, naively retraining from scratch every time a deletion request arrives is computationally prohibitive. **Machine unlearning** offers an alternative: efficiently making a model forget specific training data without full retraining.

Despite growing interest in this area, there is no widely adopted, open benchmarking framework that applies multiple unlearning methods to the same medical imaging models and forgetting scenarios, and compares them fairly on a common set of metrics. Existing work tends to evaluate one method on a specific dataset for the purposes of a single paper.

**MedForget-bench fills this gap** by providing a reusable, extensible evaluation harness — so that new unlearning methods, datasets, and metrics can be plugged in as the field evolves.

---

## What This Project Does

MedForget-bench is a modular, config-driven benchmarking framework that:

- Loads publicly available medical imaging datasets via [MedMNIST](https://medmnist.com/)
- Defines **forget scenarios** (e.g. forget all samples of a given class, or a random subset)
- Trains a reference classifier as a baseline
- Applies multiple **unlearning methods** and evaluates each one on:
  - **Utility** — does the model still perform well on retained data?
  - **Forgetting quality** — has it actually forgotten the target data?
  - **Privacy (MIA)** — does a membership inference attack still succeed on forgotten samples?
  - **Efficiency** — how much compute did this cost relative to retraining from scratch?
- Aggregates results across multiple random seeds into a reproducible comparison table and plots

Everything is built around four **plugin interfaces** (dataset / scenario / method / metric), so new components can be added without modifying the core pipeline.

---

## Positioning Against Related Work

| Work | Focus | Limitation vs. MedForget-bench |
|---|---|---|
| Nasirigerdeh et al. (2024) — *Machine Unlearning for Medical Imaging* | Empirical comparison on TissueMNIST & CheXpert | Single paper, not an open extensible harness |
| Wu et al. (2024) — *MedForget* (arXiv:2512.09867) | Unlearning for multimodal LLMs, hospital hierarchy | Targets vision-language models, not image classifiers |
| MU-Bench, Deep Unlearn | General-purpose unlearning benchmarks | Not specific to medical imaging |
| AMNESIA | Medical unlearning benchmark | Fixed method set, not open/extensible |

MedForget-bench is the first **open, extensible, classification-focused** evaluation harness for machine unlearning in the medical imaging domain.

---

## Architecture Overview

```
configs/            ← YAML experiment configs (dataset · method · scenario · seeds)
medforget/
├── datasets/       ← Dataset plugins  (implement BaseMedForgetDataset)
├── models/         ← Model wrappers   (ResNet-18 for PathMNIST)
├── scenarios/      ← Forget scenarios (implement BaseForgetScenario)
├── methods/        ← Unlearning methods (implement BaseUnlearningMethod)
├── metrics/        ← Evaluation metrics (implement BaseMetric)
├── runner.py       ← Orchestrates one full experiment from a config file
└── report.py       ← Aggregates results → comparison table + plots
scripts/            ← CLI entry points (train · run · report)
tests/              ← pytest sanity checks per plugin
notebooks/          ← Exploratory analysis
results/            ← Per-run JSON outputs (gitignored)
```

The four plugin interfaces are the architectural core: adding a new dataset, unlearning method, or metric means writing a single new file that implements one interface — the runner and report generator stay unchanged.

---

## Scope

### Initial Release

| Component | Detail |
|---|---|
| Dataset | PathMNIST (colon pathology, 9 classes, 28×28 RGB) via MedMNIST |
| Forget scenarios | Class-wise forgetting (primary) · Random-subset forgetting (secondary) |
| Unlearning methods | Exact retraining (gold-standard baseline) · Naive fine-tuning · Gradient Ascent |
| Evaluation metrics | Retain accuracy · Test accuracy · Forget-set accuracy · Loss-threshold MIA · Wall-clock efficiency |
| Statistical rigor | ≥ 3 random seeds per method/scenario — results as mean ± std |
| Model architecture | ResNet-18 (torchvision), adapted for 3-channel 28×28 input |
| Output | Reproducible comparison table (CSV + Markdown) · Matplotlib comparison plots |

### Planned Extensions

| Component | Planned Addition |
|---|---|
| Datasets | CheXpert (large-scale chest X-ray) · additional MedMNIST variants |
| Forget scenarios | Patient-wise forgetting · Feature/concept unlearning |
| Unlearning methods | SCRUB · SalUn · Fisher/influence-based unlearning · SISA · Hierarchical dual-strategy unlearning |
| Privacy metrics | Shadow-model membership inference attack (stronger than loss-threshold) |
| Efficiency metrics | Full compute and memory profiling (FLOPs, peak RAM) |
| Software | Interactive results dashboard · pip-installable package release |

The plugin architecture is specifically designed so that each planned addition requires writing one new file implementing an existing interface — no changes to the core pipeline.

---


## Setup

```bash
git clone https://github.com/rajaomar99/MedForget-bench.git
cd MedForget-bench

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt

# Verify environment
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> MedMNIST datasets download automatically on first use (~100 MB).  
> No local GPU required for setup. Training runs can be executed on Google Colab or Kaggle Notebooks.

---

## Usage

*Full CLI commands will be documented here as each component is implemented.*

---

## Reproducibility

- All experiments are fully specified by a single YAML config file
- Random seeds are fixed and logged per run
- Library versions are pinned in `requirements.txt`
- Results are saved as JSON per run and can be regenerated from the config

---

## Related Papers

- Nasirigerdeh, Razmi & Schnabel — *Machine Unlearning for Medical Imaging*, arXiv:2407.07539 (2024)
- Yang et al. — *MedMNIST v2*, Scientific Data (2023)
- ElBedoui et al. — *SoK: Federated Learning and Unlearning for Medical Image Analysis*, Expert Systems (2025)
- Hardan et al. — *Forget-MI: Machine Unlearning for Forgetting Multimodal Information in Healthcare Settings*, arXiv:2506.23145 (2025)

---

## License

MIT — open for research use and extension.
