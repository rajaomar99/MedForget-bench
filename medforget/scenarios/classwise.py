"""
medforget/scenarios/classwise.py
---------------------------------
Concrete implementation of BaseForgetScenario for class-wise forgetting.

The entire training set is partitioned into two disjoint, exhaustive subsets:
  - forget_set : all samples whose label == forget_class
  - retain_set : all remaining samples

No data is copied; both subsets are torch.utils.data.Subset objects that
simply store which indices of the original dataset to expose.
"""

from typing import Tuple

import numpy as np
from torch.utils.data import Dataset, Subset

from medforget.scenarios.base import BaseForgetScenario


class ClasswiseForgetScenario(BaseForgetScenario):
    """Forget all training samples belonging to one class.

    Args:
        forget_class: Integer class index to forget.
                      For PathMNIST, class 3 = 'lymphocytes' (default).
    """

    def __init__(self, forget_class: int = 3):
        self._forget_class = forget_class

    # ------------------------------------------------------------------
    # BaseForgetScenario interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "classwise"

    @property
    def forget_class(self) -> int:
        """The class index being forgotten."""
        return self._forget_class

    def apply(self, train_dataset: Dataset) -> Tuple[Subset, Subset]:
        """Split train_dataset into (forget_set, retain_set) by class label.

        Args:
            train_dataset: Full training dataset (e.g. from
                           PathMNISTDataset.get_train_dataset()).

        Returns:
            forget_set: Subset containing only samples of forget_class.
            retain_set: Subset containing all other samples.

        Raises:
            AssertionError: If the split is not disjoint or not exhaustive
                            (should never happen; present as a safety check).
        """
        # ── 1. Collect all labels ─────────────────────────────────────────────
        # Fast path: medmnist datasets expose a .labels array of shape (N, 1).
        # Fallback: iterate through the dataset (works for any Dataset subclass).
        if hasattr(train_dataset, "labels"):
            labels = np.array(train_dataset.labels).squeeze()   # (N, 1) → (N,)
        else:
            labels = np.array(
                [int(train_dataset[i][1]) for i in range(len(train_dataset))]
            )

        # ── 2. Partition indices ──────────────────────────────────────────────
        forget_indices = np.where(labels == self._forget_class)[0].tolist()
        retain_indices = np.where(labels != self._forget_class)[0].tolist()

        # ── 3. Invariant checks ───────────────────────────────────────────────
        assert len(set(forget_indices) & set(retain_indices)) == 0, \
            "ClasswiseForgetScenario: forget / retain sets overlap — BUG!"
        assert len(forget_indices) + len(retain_indices) == len(train_dataset), \
            "ClasswiseForgetScenario: split is not exhaustive — BUG!"

        forget_set = Subset(train_dataset, forget_indices)
        retain_set = Subset(train_dataset, retain_indices)

        return forget_set, retain_set

