"""
medforget/metrics/base.py
-------------------------
Abstract base class for all evaluation metrics in MedForget-bench.
"""

from abc import ABC, abstractmethod

import torch.nn as nn
from torch.utils.data import Dataset


class BaseMetric(ABC):
    """Contract that every evaluation metric plugin must satisfy.

    The runner calls metric.compute(...) after each unlearning run and
    receives back a single float — the score for that run on that seed.
    Statistical rigor (mean ± std) is computed at the runner level by
    aggregating scores across multiple seeds; individual metrics just
    return one number per call.

    All four dataset splits are passed so every metric can pick what
    it needs:
      - Retain accuracy    → retain_dataset
      - Test-set accuracy  → test_dataset
      - Forget accuracy    → forget_dataset
      - MIA                → forget_dataset + retain_dataset
      - Efficiency timing  → all datasets (full pipeline timing)
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, lowercase identifier used in config files and logs.

        Examples: 'retain_acc', 'test_acc', 'forget_acc', 'mia', 'efficiency'
        """

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        """Whether a higher score means better unlearning for this metric.

        True  → retain_acc, test_acc  (higher = model still useful)
        False → forget_acc, mia       (lower  = model has truly forgotten)
        """

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(
        self,
        model: nn.Module,
        forget_dataset: Dataset,
        retain_dataset: Dataset,
        test_dataset: Dataset,
    ) -> float:
        """Evaluate the unlearned model and return a scalar score.

        Args:
            model:          The unlearned model to evaluate.
            forget_dataset: Samples the model was asked to forget.
            retain_dataset: Samples the model was asked to keep.
            test_dataset:   Held-out test set (never seen during training
                            or unlearning) — used for test-set accuracy.

        Returns:
            A single float representing this metric's score for one run.
            The runner accumulates these across seeds and reports mean ± std.
        """
