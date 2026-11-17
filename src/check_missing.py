
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Load labels
with open("data/raw/labels.geojson") as f:
    gj = json.load(f)

df = pd.DataFrame([ft["properties"] for ft in gj["features"]])

# Match labels to feature files
files = {p.name for p in Path("features/arrays").glob("*.h5")}

df["fname"] = (
    df["index"].astype(str)
    + "_"
    + df["dataset"].astype(str)
    + ".h5"
)

df["has_feat"] = df["fname"].isin(files)

# Select usable multiclass samples
multi = df[df["classification_label"].notna()]
usable = multi[multi["has_feat"]]

print("Usable multiclass samples:", len(usable))

# Randomly select up to 1,000 samples
sample = usable.sample(min(1000, len(usable)), random_state=0)

# Load HDF5 arrays
arrs = []

for fn in sample["fname"]:
    with h5py.File(Path("features/arrays") / fn, "r") as h:
        arrs.append(h["array"][:])

X = np.stack(arrs)

print("\nStacked shape:", X.shape)

bands = [
    "VV", "VH", "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B9", "B11", "B12", "temp", "precip",
    "elev", "slope", "NDVI"
]

print("\nBand   NaN%   zero%   min      max")

for i, band in enumerate(bands):
    col = X[:, :, i]

    print(
        f"{band:6s}"
        f"{np.isnan(col).mean() * 100:5.1f} "
        f"{np.mean(col == 0) * 100:6.1f} "
        f"{np.nanmin(col):9.2f} "
        f"{np.nanmax(col):9.2f}"
    )

# Check whether entire sensor timesteps are zero
s1_zero = (X[:, :, 0:2] == 0).all(axis=2).mean(axis=0)
s2_zero = (X[:, :, 2:13] == 0).all(axis=2).mean(axis=0)

print(
    "\nPer-month fraction of all-zero S1:",
    np.round(s1_zero, 3)
)

print(
    "\nPer-month fraction of all-zero S2:",
    np.round(s2_zero, 3)
)
