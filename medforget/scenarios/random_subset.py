import numpy as np
from torch.utils.data import Dataset, Subset

from medforget.scenarios.base import BaseForgetScenario


class RandomSubsetForgetScenario(BaseForgetScenario):
    """Randomly selects a percentage of the entire dataset to forget.
    
    This scenario simulates a uniform unlearning request where the data to be
    forgotten is not concentrated in a single class, but spread evenly across
    the entire dataset.

    Args:
        forget_ratio: The fraction of the dataset to forget (e.g. 0.1 for 10%).
    """

    def __init__(self, forget_ratio: float = 0.1):
        if not (0.0 < forget_ratio < 1.0):
            raise ValueError(f"forget_ratio must be between 0.0 and 1.0, got {forget_ratio}")
        self._forget_ratio = forget_ratio

    @property
    def name(self) -> str:
        return "random_subset"

    def apply(self, dataset: Dataset) -> tuple[Subset, Subset]:
        """Splits the dataset into a random forget set and retain set.
        
        Args:
            dataset: The full training dataset.
            
        Returns:
            A tuple of (forget_dataset, retain_dataset).
        """
        total_len = len(dataset)
        indices = np.arange(total_len)
        
        # We rely on the global seed being set before this is called
        # to ensure reproducibility across different runs with the same seed.
        np.random.shuffle(indices)

        forget_size = int(total_len * self._forget_ratio)
        forget_indices = indices[:forget_size]
        retain_indices = indices[forget_size:]

        forget_dataset = Subset(dataset, forget_indices)
        retain_dataset = Subset(dataset, retain_indices)

        return forget_dataset, retain_dataset
