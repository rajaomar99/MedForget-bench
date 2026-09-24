"""
medforget/scenarios/base.py
---------------------------
Abstract base class for all forget scenarios in MedForget-bench.
"""

from abc import ABC, abstractmethod
from typing import Tuple

from torch.utils.data import Dataset


class BaseForgetScenario(ABC):
    """Contract that every forget scenario plugin must satisfy.

    A scenario's only job is to answer:
        'Given the full training set, which samples should be forgotten
        and which should be retained?'

    The runner calls scenario.apply(train_dataset) and receives back
    two PyTorch Datasets — forget_set and retain_set — without needing
    to know whether the split was class-wise, random-subset, or anything else.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, lowercase identifier used in config files and logs.

        Example: 'classwise' or 'random_subset'
        """

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    @abstractmethod
    def apply(self, train_dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Split train_dataset into a forget set and a retain set.

        Args:
            train_dataset: The full training dataset to partition.

        Returns:
            A tuple (forget_set, retain_set) where:
              - forget_set  : samples the model must forget
              - retain_set  : all remaining samples the model must keep
            The two sets must be disjoint and together exhaust
            train_dataset (i.e. no sample is dropped or duplicated).
        """
