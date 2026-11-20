from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd


SPLIT_CSV = Path("data/splits/optimized_spatial_seed42_cell1.csv")
FEATURE_DIR = Path("features/arrays")
OUT = Path("data/processed/cache_seed42_cell1.npz")

BANDS = [
    "VV", "VH",
    "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B11", "B12"
]

EXPECTED_SHAPE = (12, 18)


def main():
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_CSV}")

    if not FEATURE_DIR.exists():
        raise FileNotFoundError(f"Feature directory not found: {FEATURE_DIR}")

    df = pd.read_csv(SPLIT_CSV)

    required = {"fname", "classification_label", "split"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")

    if df["fname"].duplicated().any():
        raise ValueError("Duplicate filenames found in split CSV.")

    if df[["fname", "classification_label", "split"]].isna().any().any():
        raise ValueError("Missing values found in required split columns.")

    classes = sorted(df["classification_label"].unique())
    class_to_id = {name: i for i, name in enumerate(classes)}

    y = (
        df["classification_label"]
        .map(class_to_id)
        .to_numpy(dtype=np.int64)
    )

    n = len(df)
    X = np.empty((n, 12, 13), dtype=np.float32)

    bad_shape = []
    missing_files = []
    nonfinite_files = []

    print(f"Loading {n:,} samples...")

    for i, filename in enumerate(df["fname"]):
        path = FEATURE_DIR / filename

        if not path.is_file():
            missing_files.append(filename)
            continue

        try:
            with h5py.File(path, "r") as h:
                if "array" not in h:
                    bad_shape.append((filename, "missing array dataset"))
                    continue

                a = h["array"][:]

            if a.shape != EXPECTED_SHAPE:
                bad_shape.append((filename, a.shape))
                continue

            if not np.isfinite(a).all():
                nonfinite_files.append(filename)
                continue

            X[i] = a[:, :13].astype(np.float32)

        except Exception as exc:
            bad_shape.append((filename, str(exc)))

        if (i + 1) % 2000 == 0:
            print(f"Processed {i + 1:,}/{n:,}")

    if missing_files or bad_shape or nonfinite_files:
        print("\nDATA VALIDATION FAILED")
        print("Missing files:", len(missing_files), missing_files[:10])
        print("Shape/dataset problems:", len(bad_shape), bad_shape[:10])
        print("Non-finite files:", len(nonfinite_files), nonfinite_files[:10])
        sys.exit(1)

    print("\nAll files passed shape and finite-value checks.")

    # Full-dataset missingness audit.
    print("\n--- Observation audit ---")

    s1_zero = (X[:, :, 0:2] == 0).all(axis=2)
    s2_zero = (X[:, :, 2:13] == 0).all(axis=2)

    print(
        "Samples with all-zero S1 months:",
        int(s1_zero.sum())
    )
    print(
        "Samples with all-zero S2 months:",
        int(s2_zero.sum())
    )

    print(
        "S1 all-zero month fraction:",
        np.round(s1_zero.mean(axis=0), 4)
    )
    print(
        "S2 all-zero month fraction:",
        np.round(s2_zero.mean(axis=0), 4)
    )

    # Feature distribution audit.
    print("\n--- Feature statistics ---")

    for j, name in enumerate(BANDS):
        values = X[:, :, j]
        print(
            f"{name:4s} "
            f"min={values.min():10.3f} "
            f"p1={np.percentile(values, 1):10.3f} "
            f"median={np.median(values):10.3f} "
            f"p99={np.percentile(values, 99):10.3f} "
            f"max={values.max():10.3f}"
        )

    # Training-only normalization statistics.
    train_mask = df["split"].to_numpy() == "train"

    if not train_mask.any():
        raise ValueError("No training samples found.")

    mean = X[train_mask].mean(axis=(0, 1)).astype(np.float32)
    std = X[train_mask].std(axis=(0, 1)).astype(np.float32)

    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Invalid normalization statistics.")

    if (std < 1e-6).any():
        raise ValueError("One or more bands have near-zero training standard deviation.")

    print("\n--- Normalization statistics (training only) ---")
    for name, mu, sigma in zip(BANDS, mean, std):
        print(f"{name:4s} mean={mu:12.5f} std={sigma:12.5f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUT,
        X=X,
        y=y,
        split=df["split"].to_numpy(dtype=str),
        fname=df["fname"].to_numpy(dtype=str),
        classes=np.asarray(classes, dtype=str),
        bands=np.asarray(BANDS, dtype=str),
        mean=mean,
        std=std,
    )

    print("\n--- Cache saved ---")
    print("Path:", OUT)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Classes:", len(classes))
    print("Train:", int((df["split"] == "train").sum()))
    print("Validation:", int((df["split"] == "val").sum()))
    print("Test:", int((df["split"] == "test").sum()))


if __name__ == "__main__":
    main()