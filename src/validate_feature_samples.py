
from pathlib import Path
import json
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]

LABELS = ROOT / "data/raw/labels.geojson"
ARRAYS = ROOT / "features/arrays"

with open(LABELS, encoding="utf-8") as f:
    geojson = json.load(f)

records = [feature["properties"] for feature in geojson["features"]]

# Build lookup using the actual GeoJSON identifiers.
metadata = {
    (str(int(row["index"])), str(row["dataset"]))
    for row in records
}

files = sorted(ARRAYS.glob("*.h5"))

matched = 0
unmatched = []

for path in files:
    sample_id, separator, dataset = path.stem.partition("_")

    if separator and (sample_id, dataset) in metadata:
        matched += 1
    else:
        unmatched.append(path.name)

print("========== FEATURE-LABEL MAPPING ==========")
print("Total HDF5 files:", len(files))
print("Matching files:", matched)
print("Unmatched files:", len(unmatched))
print("Unmatched examples:", unmatched[:10])

# Inspect 25 files distributed across the archive.
positions = np.linspace(
    0, len(files) - 1, min(25, len(files)), dtype=int
)

nan_total = 0
inf_total = 0
zero_total = 0
element_total = 0
shapes = set()

for position in positions:
    path = files[position]

    with h5py.File(path, "r") as h5:
        array = h5["array"][:]

    shapes.add(array.shape)
    nan_total += int(np.isnan(array).sum())
    inf_total += int(np.isinf(array).sum())
    zero_total += int(np.sum(array == 0))
    element_total += array.size

print("\n========== SAMPLE MISSINGNESS ==========")
print("Files inspected:", len(positions))
print("Observed shapes:", shapes)
print("NaN values:", nan_total)
print("Infinity values:", inf_total)
print("Zero values:", zero_total)
print("Total elements:", element_total)

print("\nSample validation completed.")
