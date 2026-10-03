"""
medforget/metrics/forgetting.py
--------------------------------
Forget-set accuracy metric.

Measures how accurately the unlearned model still classifies samples from
the forget set.  Lower is better — an ideal unlearned model should be no
more accurate on forget samples than random chance (1/num_classes ≈ 0.111
for PathMNIST).

The gold-standard reference is Exact Retraining, which achieves chance-level
forget accuracy because it was never trained on those samples at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from medforget.metrics.base import BaseMetric
from medforget.metrics.utility import _top1_accuracy


class ForgetAccuracy(BaseMetric):
    """Top-1 accuracy of the unlearned model on the forget set.

    Lower is better.  Target: close to 1/num_classes (chance level).
    For PathMNIST (9 classes) that is ≈ 0.111.

    If this metric stays high after unlearning, the model has not truly
    forgotten — it can still classify forget-set samples correctly, meaning
    their influence remains embedded in the weights.

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
        return "forget_acc"

    @property
    def higher_is_better(self) -> bool:
        return False   # lower forget accuracy = better forgetting

    def compute(
        self,
        model: nn.Module,
        forget_dataset: Dataset,
        retain_dataset: Dataset,
        test_dataset: Dataset,
    ) -> float:
        """Return top-1 accuracy on the forget set.

        Args:
            model:          The unlearned model to evaluate.
            forget_dataset: Samples the model was asked to forget.
            retain_dataset: Not used by this metric.
            test_dataset:   Not used by this metric.

        Returns:
            Float in [0, 1].  Target: ≈ 1/num_classes (chance level).
            Exact Retraining achieves this by construction.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _top1_accuracy(
            model.to(device), forget_dataset,
            self._batch_size, self._num_workers, device,
        )
