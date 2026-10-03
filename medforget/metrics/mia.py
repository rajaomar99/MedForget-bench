"""
medforget/metrics/mia.py
-------------------------
Membership Inference Attack (MIA) evaluation metric.

This implements a loss-threshold attack. A model that has successfully
forgotten a dataset should not be able to distinguish between forgotten
samples (members) and held-out samples (non-members) based on its loss.

We use the test set as the non-member reference pool to see if the model 
treats the forget set differently than completely unseen data. A success 
rate close to 0.5 means the attack is guessing at chance level, indicating
successful forgetting.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from medforget.metrics.base import BaseMetric


def _compute_losses(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    """Compute per-sample cross-entropy loss for a dataset."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze(1).long().to(device)

            logits = model(imgs)
            losses = criterion(logits, labels)
            all_losses.append(losses.cpu().numpy())

    if not all_losses:
        return np.array([])
    return np.concatenate(all_losses)


class LossThresholdMIA(BaseMetric):
    """Loss-threshold Membership Inference Attack success rate.

    Lower is better. A success rate of 0.5 indicates random chance
    (perfect forgetting). Values significantly above 0.5 indicate
    the model still remembers the forget set.

    Args:
        batch_size: Inference batch size.
        num_workers: DataLoader workers (keep 0 on Windows).
    """

    def __init__(self, batch_size: int = 256, num_workers: int = 0):
        self._batch_size = batch_size
        self._num_workers = num_workers

    # ------------------------------------------------------------------
    # BaseMetric interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mia"

    @property
    def higher_is_better(self) -> bool:
        return False   # closer to 0.5 is the goal, higher than 0.5 is bad

    def compute(
        self,
        model: nn.Module,
        forget_dataset: Dataset,
        retain_dataset: Dataset,
        test_dataset: Dataset,
    ) -> float:
        """Compute the MIA success rate.

        We sample a subset of retain_dataset equal in size to forget_dataset
        to act as non-members.

        Args:
            model:          The unlearned model to evaluate.
            forget_dataset: Members of the attack.
            retain_dataset: Sampled to form non-members of the attack.
            test_dataset:   Not used by this metric.

        Returns:
            Float in [0, 1]. Target is 0.5.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # ── 1. Members ────────────────────────────────────────────────────────
        member_losses = _compute_losses(
            model, forget_dataset, self._batch_size, self._num_workers, device
        )

        # ── 2. Non-Members ────────────────────────────────────────────────────
        num_members = len(forget_dataset)
        num_test = len(test_dataset)

        # We need an equal number of non-members for a balanced 50/50 attack
        # We sample from the test_dataset (unseen data) as true non-members
        num_sample = min(num_members, num_test)
        if num_sample == 0:
            return 0.5

        # Randomly sample from test dataset
        indices = np.random.choice(num_test, num_sample, replace=False)
        sampled_test_dataset = Subset(test_dataset, indices.tolist())

        non_member_losses = _compute_losses(
            model, sampled_test_dataset, self._batch_size, self._num_workers, device
        )

        if len(member_losses) == 0 or len(non_member_losses) == 0:
            return 0.5

        # ── 3. Threshold and Attack ───────────────────────────────────────────
        # Threshold at the midpoint between medians
        member_median = np.median(member_losses)
        non_member_median = np.median(non_member_losses)
        threshold = (member_median + non_member_median) / 2.0

        # Attack logic:
        # If loss < threshold  -> guess "Member"
        # If loss >= threshold -> guess "Non-Member"
        true_positives = np.sum(member_losses < threshold)
        true_negatives = np.sum(non_member_losses >= threshold)

        success_rate = (true_positives + true_negatives) / (len(member_losses) + len(non_member_losses))

        return float(success_rate)
