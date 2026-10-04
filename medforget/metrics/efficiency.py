"""
medforget/metrics/efficiency.py
--------------------------------
Efficiency metric (wall-clock timing ratio).

Measures how much faster an approximate unlearning method is compared
to simply retraining from scratch on the retain set (Exact Retraining).

Efficiency Ratio = Time(method) / Time(ExactRetrain)

Lower is better. Values < 1.0 mean the method is faster than retraining.
Exact Retraining will have a ratio of 1.0 by definition.

Note: This metric deviates slightly from the BaseMetric signature because
it doesn't evaluate the model itself, but rather takes timing floats
provided by the runner.
"""

from __future__ import annotations

import torch.nn as nn
from torch.utils.data import Dataset

from medforget.metrics.base import BaseMetric


class EfficiencyRatio(BaseMetric):
    """Ratio of unlearning time to full retraining time.
    
    Lower is better.
    """

    # ------------------------------------------------------------------
    # BaseMetric interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "efficiency"

    @property
    def higher_is_better(self) -> bool:
        return False   # lower ratio = faster = better

    def compute(
        self,
        method_time: float = 0.0,
        retrain_time: float = 1.0,
        **kwargs
    ) -> float:
        """Compute the efficiency ratio.

        Args:
            method_time: Wall-clock seconds taken by the unlearning method.
            retrain_time: Wall-clock seconds taken by Exact Retraining.
            **kwargs: Absorbs unused standard BaseMetric arguments (model,
                      forget_dataset, retain_dataset, test_dataset) passed
                      by the runner.

        Returns:
            Ratio as a float. 
        """
        if retrain_time <= 0:
            return 0.0
            
        return method_time / retrain_time
