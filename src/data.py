import numpy as np
import torch
from torch.utils.data import Dataset


KINDS = ("random", "block", "no_s1", "no_s2")


def load_cache(path):
    """Load the cache and close the NPZ file safely."""
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def normalize(X, mean, std):
    """Normalize using training-only statistics stored in the cache."""
    X = np.asarray(X, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    if X.ndim != 3 or X.shape[1:] != (12, 13):
        raise ValueError(f"Expected X shape (N, 12, 13), got {X.shape}")

    if mean.shape != (13,) or std.shape != (13,):
        raise ValueError("mean and std must each have shape (13,)")

    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Normalization statistics contain NaN or infinity")

    if (std <= 0).any():
        raise ValueError("Standard deviations must be positive")

    return ((X - mean[None, None, :]) / std[None, None, :]).astype(
        np.float32
    )


def _validate_mask_inputs(kind, rate):
    if kind not in KINDS:
        raise ValueError(f"Unknown mask kind: {kind}. Choose from {KINDS}")

    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0.0 and 1.0")


def _one_mask(kind, rate, rng):
    """
    Create a (12, 2) boolean mask.

    True = observed
    False = missing
    Column 0 = Sentinel-1
    Column 1 = Sentinel-2
    """
    _validate_mask_inputs(kind, rate)

    mask = np.ones((12, 2), dtype=bool)

    if kind == "random":
        mask = rng.random((12, 2)) >= rate

    elif kind == "block":
        # Remove one contiguous block of Sentinel-2 months.
        length = int(round(rate * 12))

        if length > 0:
            start = int(rng.integers(0, 12 - length + 1))
            mask[start:start + length, 1] = False

    elif kind == "no_s1":
        mask[:, 0] = False

    elif kind == "no_s2":
        mask[:, 1] = False

    return mask


def make_masks(n, kind, rate, seed):
    """Generate reproducible masks for n samples."""
    if n < 0:
        raise ValueError("n cannot be negative")

    _validate_mask_inputs(kind, rate)

    masks = np.empty((n, 12, 2), dtype=bool)

    for i in range(n):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), i])
        )
        masks[i] = _one_mask(kind, rate, rng)

    return masks


def apply_mask(X, M):
    """
    Apply masks to normalized feature arrays.

    X: (N, 12, 13)
    M: (N, 12, 2)

    Missing normalized observations are replaced with zero.
    """
    X = np.asarray(X, dtype=np.float32)
    M = np.asarray(M, dtype=bool)

    if X.ndim != 3 or X.shape[1:] != (12, 13):
        raise ValueError(f"Expected X shape (N, 12, 13), got {X.shape}")

    if M.shape != (X.shape[0], 12, 2):
        raise ValueError(
            f"Expected mask shape {(X.shape[0], 12, 2)}, got {M.shape}"
        )

    result = X.copy()

    result[:, :, 0:2] *= M[:, :, 0:1]
    result[:, :, 2:13] *= M[:, :, 1:2]

    return result


class CropDataset(Dataset):
    """
    PyTorch dataset.

    Pass normalized features to this class.
    Returns: (features, availability_mask, label)

    Optional augmentation:
        aug=("random", 0.5)
        aug=("block", 0.5)

    Call set_epoch(epoch) before each training epoch to change
    augmentation masks reproducibly.
    """

    def __init__(
        self,
        X,
        y,
        masks=None,
        aug=None,
        seed=42
    ):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 3 or X.shape[1:] != (12, 13):
            raise ValueError(f"Expected X shape (N, 12, 13), got {X.shape}")

        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must have shape (N,) matching X")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity")

        if masks is not None:
            masks = np.asarray(masks, dtype=bool)
            if masks.shape != (len(X), 12, 2):
                raise ValueError("Invalid masks shape")

        if aug is not None:
            if len(aug) != 2:
                raise ValueError("aug must be a (kind, max_rate) tuple")

            kind, max_rate = aug
            _validate_mask_inputs(kind, max_rate)

        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.y = torch.from_numpy(np.ascontiguousarray(y))

        self.masks = (
            None if masks is None
            else torch.from_numpy(np.ascontiguousarray(masks))
        )

        self.aug = aug
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.masks is not None:
            mask = self.masks[i]

        elif self.aug is not None:
            kind, max_rate = self.aug

            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, int(i)])
            )

            rate = float(rng.uniform(0.0, max_rate))
            generated = _one_mask(kind, rate, rng)
            mask = torch.from_numpy(generated)

        else:
            mask = torch.ones((12, 2), dtype=torch.bool)

        x = self.X[i].clone()

        x[:, 0:2] *= mask[:, 0:1]
        x[:, 2:13] *= mask[:, 1:2]

        return x, mask, self.y[i]