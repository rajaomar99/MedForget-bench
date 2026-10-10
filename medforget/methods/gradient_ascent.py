"""
medforget/methods/gradient_ascent.py
------------------------------------
Gradient Ascent with Retain-set Regularization (Push-Pull method).

This method actively forces the model to unlearn the forget set by
maximizing its loss (Gradient Ascent). To prevent catastrophic
forgetting of the entire model, it alternates by minimizing the loss
on the retain set (Gradient Descent).
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from medforget.methods.base import BaseUnlearningMethod


class GradientAscent(BaseUnlearningMethod):
    """Gradient Ascent unlearning with retain-set regularization.

    Args:
        epochs:      How many epochs to unlearn for (default: 5).
        lr:          Learning rate. Usually smaller than normal training.
        batch_size:  Training batch size.
        num_workers: DataLoader workers (0 for Windows compatibility).
    """

    def __init__(
        self,
        epochs: int = 5,
        lr: float = 1e-4,  # GA often requires a smaller LR to prevent explosions
        batch_size: int = 256,
        num_workers: int = 0,
    ):
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._num_workers = num_workers

    @property
    def name(self) -> str:
        return "gradient_ascent"

    def run(
        self,
        model: nn.Module,
        forget_dataset: Dataset,
        retain_dataset: Dataset,
    ) -> nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create an independent copy so we don't mutate the original baseline model
        unlearned_model = copy.deepcopy(model).to(device)
        unlearned_model.train()

        # We need loaders for BOTH datasets now
        forget_loader = DataLoader(
            forget_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )
        
        retain_loader = DataLoader(
            retain_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

        optimizer = torch.optim.Adam(unlearned_model.parameters(), lr=self._lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self._epochs):
            # We iterate over the larger retain set, and loop the forget set as needed.
            # This ensures we get plenty of "repair" (descent) steps to keep the model healthy.
            
            pbar = tqdm(retain_loader, desc=f"GradientAscent Epoch {epoch+1}/{self._epochs}", leave=False)
            forget_iter = iter(forget_loader)
            
            for retain_imgs, retain_labels in pbar:
                retain_imgs = retain_imgs.to(device)
                retain_labels = retain_labels.squeeze(1).long().to(device)

                # 1. Gradient Ascent on Forget Set (Push)
                try:
                    forget_imgs, forget_labels = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_imgs, forget_labels = next(forget_iter)
                    
                forget_imgs = forget_imgs.to(device)
                forget_labels = forget_labels.squeeze(1).long().to(device)

                optimizer.zero_grad()
                forget_logits = unlearned_model(forget_imgs)
                forget_loss = criterion(forget_logits, forget_labels)
                
                # FLIP: Negative loss maximizes the error
                ascent_loss = -forget_loss 
                ascent_loss.backward()
                optimizer.step()

                # 2. Gradient Descent on Retain Set (Pull/Repair)
                optimizer.zero_grad()
                retain_logits = unlearned_model(retain_imgs)
                retain_loss = criterion(retain_logits, retain_labels)
                
                retain_loss.backward()
                optimizer.step()

                pbar.set_postfix({
                    "f_loss": f"{forget_loss.item():.4f}", 
                    "r_loss": f"{retain_loss.item():.4f}"
                })

        unlearned_model.eval()
        return unlearned_model
