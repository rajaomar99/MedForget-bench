"""
medforget/metrics/utility.py
-----------------------------
Utility metrics: retain-set accuracy and held-out test-set accuracy.

Both measure whether the unlearned model still correctly classifies samples
it was supposed to keep.  Higher accuracy = better utility preservation.

These are the "do no harm" metrics — a large drop here means the unlearning
method caused collateral damage on data it was not supposed to touch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from medforget.metrics.base import BaseMetric


# ── Shared helper ─────────────────────────────────────────────────────────────

def _top1_accuracy(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> float:
    """Compute top-1 classification accuracy of *model* on *dataset*.

    Args:
        model:       Model to evaluate (set to eval mode internally).
        dataset:     Dataset to evaluate on.
        batch_size:  Mini-batch size for the DataLoader.
        num_workers: DataLoader worker processes (0 = main-process only,
                     required on Windows to avoid multiprocessing issues).
        device:      Torch device to run inference on.

    Returns:
        Top-1 accuracy as a float in [0, 1].
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,           # order doesn't matter for accuracy
        num_workers=num_workers,
    )

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.squeeze(1).long().to(device)   # (B, 1) → (B,)

            preds   = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total if total > 0 else 0.0


# ── Metric classes ────────────────────────────────────────────────────────────

class RetainAccuracy(BaseMetric):
    """Top-1 accuracy of the unlearned model on the retain set.

    Higher is better.  The target is to stay within ~5 percentage points
    of the pre-unlearning baseline — any larger drop indicates catastrophic
    forgetting of data the method was supposed to preserve.

    Args:
        batch_size:  Inference batch size (larger = faster on GPU).
        num_workers: DataLoader workers (keep 0 on Windows).
    """

    def __init__(self, batch_size: int = 256, num_workers: int = 0):
        self._batch_size  = batch_size
        self._num_workers = num_workers

    # ------------------------------------------------------------------
    # BaseMetric interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "retain_acc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute(
        self,
        model: nn.Module,
        forget_dataset: Dataset,
        retain_dataset: Dataset,
        test_dataset: Dataset,
    ) -> float:
        """Return top-1 accuracy on the retain set.

        Args:
            model:          The unlearned model to evaluate.
            forget_dataset: Not used by this metric.
            retain_dataset: All training samples the model should still know.
            test_dataset:   Not used by this metric.

        Returns:
            Float in [0, 1].  Aim for close to pre-unlearning baseline.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _top1_accuracy(
            model.to(device), retain_dataset,
            self._batch_size, self._num_workers, device,
        )


class TestAccuracy(BaseMetric):
    """Top-1 accuracy of the unlearned model on the held-out test set.

    Higher is better.  Measures generalisation after unlearning — the model
    should still classify unseen examples correctly.

    Reference: MedMNIST ResNet-18 baseline on PathMNIST ≈ 0.88 top-1.

    Args:
        batch_size:  Inference batch size.
        num_workers: DataLoader workers (keep 0 on Windows).
    """

    def __init__(self, batch_size: int = 256, num_workers: int = 0):
        self._batch_size  = batch_size
        self._num_workers = num_workers

    # ------------------------------------------------------------------
    # BaseMetric interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "test_acc"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute(
        self,
        model: nn.Module,
        forget_dataset: Dataset,
        retain_dataset: Dataset,
        test_dataset: Dataset,
    ) -> float:
        """Return top-1 accuracy on the held-out test set.

        Args:
            model:          The unlearned model to evaluate.
            forget_dataset: Not used by this metric.
            retain_dataset: Not used by this metric.
            test_dataset:   The held-out set (never seen during training or
                            unlearning) — used to check generalisation.

        Returns:
            Float in [0, 1].  Should remain close to ~0.88 after unlearning.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _top1_accuracy(
            model.to(device), test_dataset,
            self._batch_size, self._num_workers, device,
        )
