"""
tests/test_methods.py
---------------------
Pytest sanity tests for unlearning methods.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

from medforget.methods.exact_retrain import ExactRetrain
from medforget.methods.finetune import NaiveFineTune


class DummyModel(nn.Module):
    """A tiny model for fast testing."""
    def __init__(self, num_classes=9):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 28 * 28, num_classes)
        
    def forward(self, x):
        return self.fc(self.flatten(x))


class DummyDataset(Dataset):
    """A random dataset for testing training loops."""
    def __init__(self, size=32, num_classes=9):
        self.data = torch.randn(size, 3, 28, 28)
        self.labels = torch.randint(0, num_classes, (size, 1))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@pytest.fixture
def dummy_data():
    full_ds = DummyDataset(size=128)
    # Split into 32 forget, 96 retain
    forget_ds = Subset(full_ds, range(0, 32))
    retain_ds = Subset(full_ds, range(32, 128))
    return forget_ds, retain_ds


class TestMethods:
    def test_exact_retrain(self, dummy_data):
        forget_ds, retain_ds = dummy_data
        
        def model_factory():
            return DummyModel()
            
        method = ExactRetrain(model_factory=model_factory, epochs=1, batch_size=16, num_workers=0)
        baseline = DummyModel()
        
        # ExactRetrain ignores baseline, but interface requires it
        unlearned = method.run(baseline, forget_ds, retain_ds)
        
        assert isinstance(unlearned, nn.Module)
        assert unlearned is not baseline  # Must be a newly trained model

    def test_naive_finetune(self, dummy_data):
        forget_ds, retain_ds = dummy_data
        
        baseline = DummyModel()
        # Save original weights to check if they change
        original_weights = baseline.fc.weight.data.clone()
        
        method = NaiveFineTune(num_epochs=1, batch_size=16, num_workers=0)
        unlearned = method.run(baseline, forget_ds, retain_ds)
        
        assert isinstance(unlearned, nn.Module)
        assert unlearned is not baseline  # Must not mutate original model in-place
        
        # Verify unlearned model's weights have been updated (training happened)
        assert not torch.equal(unlearned.fc.weight.data, original_weights)
        # Verify baseline model's weights were NOT mutated
        assert torch.equal(baseline.fc.weight.data, original_weights)

    def test_gradient_ascent(self, dummy_data):
        from medforget.methods.gradient_ascent import GradientAscent
        forget_ds, retain_ds = dummy_data
        
        baseline = DummyModel()
        original_weights = baseline.fc.weight.data.clone()
        
        method = GradientAscent(epochs=1, batch_size=16, num_workers=0)
        unlearned = method.run(baseline, forget_ds, retain_ds)
        
        assert isinstance(unlearned, nn.Module)
        assert unlearned is not baseline
        assert not torch.equal(unlearned.fc.weight.data, original_weights)
        assert torch.equal(baseline.fc.weight.data, original_weights)
