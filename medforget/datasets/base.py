"""
medforget/datasets/base.py
--------------------------
Abstract base class for all datasets in MedForget-bench.
"""

from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseMedForgetDataset(ABC):
    """Contract that every dataset plugin must satisfy."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, lowercase identifier used in config files and logs.

        Example: 'pathmnist'
        """

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of target classes in this dataset.

        Example: 9 for PathMNIST.
        """

    # ------------------------------------------------------------------
    # Data splits
    # ------------------------------------------------------------------

    @abstractmethod
    def get_train_dataset(self) -> Dataset:
        """Return the full training split as a PyTorch Dataset."""

    @abstractmethod
    def get_val_dataset(self) -> Dataset:
        """Return the validation split as a PyTorch Dataset."""

    @abstractmethod
    def get_test_dataset(self) -> Dataset:
        """Return the test split as a PyTorch Dataset."""
