"""
scripts/train_baseline.py
-------------------------
Trains the baseline ResNet-18 model on the full PathMNIST training dataset.
This produces the 'original' memorized model that unlearning methods will
use as a starting point.

Usage:
    python scripts/train_baseline.py --epochs 30 --batch_size 128 --save_dir checkpoints
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from medforget.datasets.pathmnist import PathMNISTDataset
from medforget.models.resnet18 import build_resnet18


def train_one_epoch(model, loader, criterion, optimizer, device, fast_dev_run=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device)
        labels = labels.squeeze(1).long().to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        if fast_dev_run and batch_idx >= 2:
            break

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, fast_dev_run=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.squeeze(1).long().to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if fast_dev_run and batch_idx >= 2:
            break

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train Baseline PathMNIST Model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save the best model")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 2 batches per epoch for testing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "baseline.pt"

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("Loading PathMNIST dataset...")
    ds = PathMNISTDataset(download=True)
    train_set = ds.get_train_dataset()
    val_set = ds.get_val_dataset()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── 2. Model ─────────────────────────────────────────────────────────────
    print(f"Building ResNet-18 for {ds.num_classes} classes...")
    model = build_resnet18(num_classes=ds.num_classes).to(device)

    # ── 3. Optimizer & Loss ──────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── 4. Training Loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.fast_dev_run
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, args.fast_dev_run
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"--> New best validation accuracy! Saving model to {save_path}")
            torch.save(model.state_dict(), save_path)

        if args.fast_dev_run:
            print("Fast dev run enabled. Stopping after 1 epoch.")
            break

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
