
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "data" / "raw" / "labels.geojson"
ARRAYS_DIR = ROOT / "features" / "arrays"


def load_labels():
    """Load GeoJSON without assuming column names."""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(LABELS_PATH)
        return pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    except Exception as exc:
        print(f"GeoPandas unavailable or failed: {exc}")
        print("Trying plain JSON...")
        with open(LABELS_PATH, encoding="utf-8") as f:
            geojson = json.load(f)

        return pd.DataFrame(
            feature.get("properties", {})
            for feature in geojson["features"]
        )


def inspect_labels():
    print("\n========== LABELS GEOJSON ==========")

    df = load_labels()

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nNumber of rows:", len(df))

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nUnique values per column:")
    print(df.nunique(dropna=False))

    print("\nPotential class-label counts:")
    for col in df.columns:
        name = col.lower()
        unique_count = df[col].nunique(dropna=False)

        if (
            any(word in name for word in
                ["label", "class", "crop", "target"])
            or unique_count <= 50
        ):
            print(f"\nColumn: {col}")
            print(df[col].value_counts(dropna=False).head(50))


def inspect_hdf5():
    print("\n========== HDF5 ARRAY ==========")

    files = sorted(ARRAYS_DIR.glob("*.h5"))

    if not files:
        raise FileNotFoundError(
            f"No HDF5 files found in {ARRAYS_DIR}"
        )

    sample_path = files[0]
    print("\nSample file:", sample_path.name)

    with h5py.File(sample_path, "r") as h5:
        print("\nHDF5 keys:", list(h5.keys()))

        if "array" not in h5:
            raise KeyError("Expected dataset 'array' not found")

        array = h5["array"][:]

    print("\nArray shape:", array.shape)
    print("Data type:", array.dtype)

    print("\nMissing-value inspection:")
    print("NaN count:", np.isnan(array).sum())
    print("Infinity count:", np.isinf(array).sum())
    print("Zero count:", np.sum(array == 0))
    print("Total elements:", array.size)

    finite = array[np.isfinite(array)]

    if finite.size:
        print("Finite minimum:", finite.min())
        print("Finite maximum:", finite.max())

    print("\nMost frequent values:")
    print(
        pd.Series(array.ravel())
        .value_counts(dropna=False)
        .head(10)
    )


if __name__ == "__main__":
    inspect_labels()
    inspect_hdf5()
