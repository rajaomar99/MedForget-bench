"""
medforget/methods/base.py
-------------------------
Abstract base class for all unlearning methods in MedForget-bench.
"""

from abc import ABC, abstractmethod

import torch.nn as nn
from torch.utils.data import Dataset


class BaseUnlearningMethod(ABC):
    """Contract that every unlearning method plugin must satisfy.

    The runner's perspective is simple:
        'Here is a trained model, here is the forget set, here is the
        retain set — give me back a model that has forgotten the forget set.'

    How the method achieves that (retraining, fine-tuning, gradient ascent,
    etc.) is entirely internal to the subclass.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, lowercase identifier used in config files and logs.

        Examples: 'exact_retrain', 'naive_finetune', 'gradient_ascent'
        """

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        model: nn.Module,
        forget_dataset: Dataset,
        retain_dataset: Dataset,
    ) -> nn.Module:
        """Apply unlearning and return the modified model.

        Args:
            model:          The original trained model (will NOT be
                            modified in-place; implementations should
                            work on a copy if needed).
            forget_dataset: Samples the model must forget.
            retain_dataset: Samples the model must continue to perform
                            well on.

        Returns:
            The unlearned model as a PyTorch nn.Module, ready for
            evaluation by the metrics pipeline.
        """
