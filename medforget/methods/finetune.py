"""
medforget/methods/finetune.py
-----------------------------
Naive Fine-Tuning unlearning method.

This method takes a pre-trained model and continues training it
only on the retain dataset for a small number of epochs. 
It completely ignores the forget dataset.

The intuition is that by exposing the model only to the data it 
should remember, it will gradually overwrite or "forget" the 
representations of the data it is no longer seeing.
"""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from medforget.methods.base import BaseUnlearningMethod


class NaiveFineTune(BaseUnlearningMethod):
    """Naive Fine-Tuning on the retain set.

    Args:
        num_epochs:  How many epochs to fine-tune for (default: 5).
                     Requires far fewer epochs than training from scratch.
        lr:          Learning rate for Adam optimizer.
        batch_size:  Training batch size.
        num_workers: DataLoader workers (0 for Windows compatibility).
    """

    def __init__(
        self,
        num_epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 256,
        num_workers: int = 0,
    ):
        self._num_epochs = num_epochs
        self._lr = lr
        self._batch_size = batch_size
        self._num_workers = num_workers

    @property
    def name(self) -> str:
        return "naive_finetune"

    def run(
        self,
        model: nn.Module,
        forget_dataset: Dataset,
        retain_dataset: Dataset,
    ) -> nn.Module:
        """Execute naive fine-tuning.

        Args:
            model:          The pre-trained baseline model.
            forget_dataset: Ignored by this method.
            retain_dataset: The data to continue training on.

        Returns:
            A new model instance that has been fine-tuned.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create an independent copy so we don't mutate the original baseline model
        unlearned_model = copy.deepcopy(model).to(device)
        unlearned_model.train()

        loader = DataLoader(
            retain_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

        optimizer = torch.optim.Adam(unlearned_model.parameters(), lr=self._lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self._num_epochs):
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.squeeze(1).long().to(device)

                optimizer.zero_grad()
                logits = unlearned_model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        unlearned_model.eval()
        return unlearned_model
