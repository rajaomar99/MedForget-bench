"""
tests/test_metrics.py
---------------------
Pytest sanity tests for metrics.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

from medforget.metrics.utility import RetainAccuracy, TestAccuracy
from medforget.metrics.forgetting import ForgetAccuracy
from medforget.metrics.mia import LossThresholdMIA
from medforget.metrics.efficiency import EfficiencyRatio


class DummyModel(nn.Module):
    """A tiny model for fast testing."""
    def __init__(self, num_classes=9):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 28 * 28, num_classes)
        
    def forward(self, x):
        return self.fc(self.flatten(x))


class DummyDataset(Dataset):
    """A random dataset for testing metric loops."""
    def __init__(self, size=32, num_classes=9):
        self.data = torch.randn(size, 3, 28, 28)
        self.labels = torch.randint(0, num_classes, (size, 1))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@pytest.fixture
def metric_data():
    forget_ds = DummyDataset(size=32)
    retain_ds = DummyDataset(size=64)
    test_ds = DummyDataset(size=32)
    model = DummyModel()
    return model, forget_ds, retain_ds, test_ds


class TestMetrics:
    def test_retain_accuracy(self, metric_data):
        model, forget_ds, retain_ds, test_ds = metric_data
        metric = RetainAccuracy(batch_size=16, num_workers=0)
        
        score = metric.compute(model, forget_ds, retain_ds, test_ds)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_test_accuracy(self, metric_data):
        model, forget_ds, retain_ds, test_ds = metric_data
        metric = TestAccuracy(batch_size=16, num_workers=0)
        
        score = metric.compute(model, forget_ds, retain_ds, test_ds)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_forget_accuracy(self, metric_data):
        model, forget_ds, retain_ds, test_ds = metric_data
        metric = ForgetAccuracy(batch_size=16, num_workers=0)
        
        score = metric.compute(model, forget_ds, retain_ds, test_ds)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_mia(self, metric_data):
        model, forget_ds, retain_ds, test_ds = metric_data
        metric = LossThresholdMIA(batch_size=16, num_workers=0)
        
        score = metric.compute(model, forget_ds, retain_ds, test_ds)
        assert isinstance(score, float)
        # MIA on completely random weights and data should be near 0.5
        assert 0.0 <= score <= 1.0

    def test_efficiency(self):
        metric = EfficiencyRatio()
        # If method takes 5 seconds, and baseline retrain takes 10 seconds, ratio is 0.5
        score = metric.compute(method_time=5.0, retrain_time=10.0)
        assert score == 0.5
