"""
medforget/methods/exact_retrain.py
-----------------------------------
Exact-retraining baseline for machine unlearning.

The passed-in model is completely discarded.  A brand-new model is built
from scratch using model_factory(), then trained exclusively on the retain
set.  Because the forgotten samples never participate in any gradient step,
this is the theoretical gold standard — it produces a model that has
genuinely never seen the forget set.

Cost: O(full retrain on retain set) — slower than approximate methods but
serves as the upper-bound reference for all evaluation metrics.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from medforget.methods.base import BaseUnlearningMethod


class ExactRetrain(BaseUnlearningMethod):
    """Retrain a fresh model from scratch on the retain set only.

    Args:
        model_factory: Zero-argument callable that returns a freshly
                       initialised nn.Module, e.g.
                       ``lambda: build_resnet18(num_classes=9)``
        epochs:        Number of training epochs.  Default 30 is suitable
                       for PathMNIST on Colab; reduce for smoke tests.
        lr:            Learning rate for Adam.
        batch_size:    Mini-batch size.
        num_workers:   DataLoader worker processes (0 = main process only,
                       safe on Windows).
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        self._model_factory = model_factory
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._num_workers = num_workers

    # ------------------------------------------------------------------
    # BaseUnlearningMethod interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "exact_retrain"

    def run(
        self,
        model: nn.Module,          # ignored — we train from scratch
        forget_dataset,            # ignored — never touched
        retain_dataset,
    ) -> nn.Module:
        """Train a fresh model on retain_dataset and return it.

        The original ``model`` argument is intentionally unused.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Fresh model ───────────────────────────────────────────────────────
        fresh_model = self._model_factory().to(device)

        # ── Data ─────────────────────────────────────────────────────────────
        loader = DataLoader(
            retain_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

        # ── Optimiser + loss ──────────────────────────────────────────────────
        optimizer = torch.optim.Adam(fresh_model.parameters(), lr=self._lr)
        criterion = nn.CrossEntropyLoss()

        # ── Training loop ─────────────────────────────────────────────────────
        fresh_model.train()
        for epoch in range(self._epochs):
            epoch_loss = 0.0
            
            pbar = tqdm(loader, desc=f"ExactRetrain Epoch {epoch+1}/{self._epochs}", leave=False)
            for imgs, labels in pbar:
                imgs   = imgs.to(device)
                labels = labels.squeeze(1).long().to(device)  # (B,1) → (B,)

                optimizer.zero_grad()
                logits = fresh_model(imgs)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

        return fresh_model
