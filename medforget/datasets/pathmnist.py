"""
medforget/datasets/pathmnist.py
--------------------------------
Concrete implementation of BaseMedForgetDataset for PathMNIST.

PathMNIST is a 9-class colon pathology classification dataset from the
MedMNIST benchmark.  Images are 28×28 RGB patches from the NCT-CRC-HE-100K
colorectal cancer histology dataset.

Classes (0–8):
    0: adipose               5: smooth muscle
    1: background            6: normal colon mucosa
    2: debris                7: cancer-associated stroma
    3: lymphocytes           8: colorectal adenocarcinoma epithelium
    4: mucus

Reference: Yang et al., "MedMNIST v2", Scientific Data 2023.
"""

from torchvision import transforms
from medmnist import PathMNIST

from medforget.datasets.base import BaseMedForgetDataset

# ── Per-channel normalisation constants ──────────────────────────────────────
# Computed from the PathMNIST training set (89,996 samples, 28×28 RGB).
# Pixels are first scaled to [0, 1] by transforms.ToTensor(), then each
# channel is standardised to zero mean and unit variance.
_PATHMNIST_MEAN = (0.7405, 0.5330, 0.7058)   # R, G, B
_PATHMNIST_STD  = (0.1237, 0.1768, 0.1244)   # R, G, B


class PathMNISTDataset(BaseMedForgetDataset):
    """PathMNIST dataset — 28×28 RGB colon pathology images, 9 classes.

    Note on labels:
        MedMNIST returns labels as numpy arrays of shape (1,).  The
        DataLoader therefore produces label batches of shape (B, 1).
        Training / evaluation code should call .squeeze(1) on labels
        before passing them to a loss function.
    """

    def __init__(self, download: bool = True):
        """
        Args:
            download: If True, download the dataset if not already cached.
                      The cache directory is ~/.medmnist/.
        """
        self._download = download
        self._transform = transforms.Compose([
            transforms.ToTensor(),                        # PIL → float tensor [0, 1]
            transforms.Normalize(_PATHMNIST_MEAN,         # standardise each channel
                                 _PATHMNIST_STD),
        ])

    # ------------------------------------------------------------------
    # BaseMedForgetDataset interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "pathmnist"

    @property
    def num_classes(self) -> int:
        return 9

    def get_train_dataset(self) -> PathMNIST:
        return PathMNIST(
            split="train",
            transform=self._transform,
            download=self._download,
            size=28,
        )

    def get_val_dataset(self) -> PathMNIST:
        return PathMNIST(
            split="val",
            transform=self._transform,
            download=self._download,
            size=28,
        )

    def get_test_dataset(self) -> PathMNIST:
        return PathMNIST(
            split="test",
            transform=self._transform,
            download=self._download,
            size=28,
        )
