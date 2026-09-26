"""
tests/test_scenarios.py
-----------------------
Pytest sanity tests for forget scenario implementations.

Tests run against a lightweight DummyDataset so no MedMNIST download is
required.  Each test focuses on one invariant.
"""

import numpy as np
import pytest
from torch.utils.data import Dataset

from medforget.scenarios.classwise import ClasswiseForgetScenario


# ── Minimal fake dataset ──────────────────────────────────────────────────────

class DummyDataset(Dataset):
    """Fake dataset with balanced classes — no images, just labels.

    Labels are stored as shape-(N, 1) numpy arrays to match the medmnist
    format that ClasswiseForgetScenario expects.

    Args:
        num_classes:       Number of distinct class indices (default 9).
        samples_per_class: How many samples per class (default 100).
    """

    def __init__(self, num_classes: int = 9, samples_per_class: int = 100):
        n = num_classes * samples_per_class
        raw = [i // samples_per_class for i in range(n)]
        self.labels = np.array(raw, dtype=np.int64).reshape(-1, 1)  # (N, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        # Return a zero image + label (image content doesn't matter for tests)
        return np.zeros((3, 28, 28), dtype=np.float32), self.labels[idx]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_ds():
    """900-sample balanced dataset: 9 classes × 100 samples each."""
    return DummyDataset(num_classes=9, samples_per_class=100)


# ── Tests: ClasswiseForgetScenario ────────────────────────────────────────────

class TestClasswiseForgetScenario:

    def test_name(self):
        """The scenario must identify itself as 'classwise'."""
        assert ClasswiseForgetScenario().name == "classwise"

    def test_forget_class_property(self):
        """Constructor argument is exposed via the forget_class property."""
        assert ClasswiseForgetScenario(forget_class=5).forget_class == 5

    def test_disjoint(self, dummy_ds):
        """Forget and retain sets must not share any index."""
        forget_set, retain_set = ClasswiseForgetScenario(forget_class=3).apply(dummy_ds)
        overlap = set(forget_set.indices) & set(retain_set.indices)
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_exhaustive(self, dummy_ds):
        """Every training sample must appear in exactly one of the two sets."""
        forget_set, retain_set = ClasswiseForgetScenario(forget_class=3).apply(dummy_ds)
        assert len(forget_set) + len(retain_set) == len(dummy_ds)

    def test_forget_set_contains_only_target_class(self, dummy_ds):
        """All samples in the forget set must belong to forget_class."""
        forget_class = 3
        forget_set, _ = ClasswiseForgetScenario(forget_class=forget_class).apply(dummy_ds)
        for idx in forget_set.indices:
            label = dummy_ds.labels[idx].item()
            assert label == forget_class, \
                f"Found label {label} in forget set (expected {forget_class})"

    def test_retain_set_contains_no_target_class(self, dummy_ds):
        """No sample in the retain set should belong to forget_class."""
        forget_class = 3
        _, retain_set = ClasswiseForgetScenario(forget_class=forget_class).apply(dummy_ds)
        for idx in retain_set.indices:
            label = dummy_ds.labels[idx].item()
            assert label != forget_class, \
                f"Found label {label} (forget_class) in retain set"

    def test_forget_set_size(self, dummy_ds):
        """With 100 samples per class, the forget set should have exactly 100."""
        forget_set, _ = ClasswiseForgetScenario(forget_class=3).apply(dummy_ds)
        assert len(forget_set) == 100

    def test_retain_set_size(self, dummy_ds):
        """With 9 classes × 100 samples, retain should have 8 × 100 = 800."""
        _, retain_set = ClasswiseForgetScenario(forget_class=3).apply(dummy_ds)
        assert len(retain_set) == 800

    @pytest.mark.parametrize("forget_class", [0, 4, 8])
    def test_invariants_for_multiple_classes(self, dummy_ds, forget_class):
        """Disjoint + exhaustive invariants hold for any valid forget_class."""
        forget_set, retain_set = ClasswiseForgetScenario(forget_class=forget_class).apply(dummy_ds)
        overlap = set(forget_set.indices) & set(retain_set.indices)
        assert len(overlap) == 0
        assert len(forget_set) + len(retain_set) == len(dummy_ds)
