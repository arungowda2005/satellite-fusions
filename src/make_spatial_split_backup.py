import json, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import BallTree

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CELL = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0   # grid size in degrees
FRAC = {"train": 0.70, "val": 0.15, "test": 0.15}

# ---- usable multiclass samples (same logic as validate_mapping.py) ----
gj = json.load(open("data/raw/labels.geojson"))
df = pd.DataFrame([f["properties"] for f in gj["features"]])
files = {p.name for p in Path("features/arrays").glob("*.h5")}
df["fname"] = df["index"].astype(str) + "_" + df["dataset"].astype(str) + ".h5"
df = df[df["classification_label"].notna() & df["fname"].isin(files)].copy()
df["lat"] = pd.to_numeric(df["lat"]); df["lon"] = pd.to_numeric(df["lon"])
print("usable samples:", len(df), "| is_test=True among them:", int(df["is_test"].sum()))

# ---- PART A: where do classes live? ----
print("\nclass counts per source dataset:")
print(pd.crosstab(df["dataset"], df["classification_label"]))
print("\nsource datasets per class:")
print(df.groupby("classification_label")["dataset"].nunique())

# ---- PART B: grid-cell blocks, greedy class-balanced assignment ----
df["cell"] = (np.floor(df["lat"] / CELL).astype(int).astype(str) + "_" +
              np.floor(df["lon"] / CELL).astype(int).astype(str))
classes = sorted(df["classification_label"].unique())
cc = pd.crosstab(df["cell"], df["classification_label"]).reindex(columns=classes, fill_value=0)
print(f"\n{len(cc)} grid cells at {CELL} deg; largest cell holds "
      f"{cc.sum(axis=1).max() / len(df) * 100:.1f}% of samples")

total = cc.sum().values.astype(float)
target = {s: total * f for s, f in FRAC.items()}
cur = {s: np.zeros(len(classes)) for s in FRAC}
rng = np.random.default_rng(SEED)
sizes = cc.sum(axis=1)
order = (sizes * rng.uniform(0.8, 1.2, len(sizes))).sort_values(ascending=False).index

cell_split = {}
for cell in order:
    v = cc.loc[cell].values.astype(float)
    best, best_delta = None, None
    for s in FRAC:
        t = target[s] + 1.0
        delta = np.sum(((cur[s] + v - target[s]) / t) ** 2) - np.sum(((cur[s] - target[s]) / t) ** 2)
        if best_delta is None or delta < best_delta:
            best, best_delta = s, delta
    cell_split[cell] = best
    cur[best] += v

df["split"] = df["cell"].map(cell_split)

# ---- report ----
print("\nsplit sizes:")
print(df["split"].value_counts(normalize=True).round(3))
print("\nclass counts per split:")
tab = pd.crosstab(df["classification_label"], df["split"])[["train", "val", "test"]]
print(tab)
for s in ["val", "test"]:
    low = tab[s][tab[s] < 30]
    if len(low):
        print(f"WARNING: classes with <30 samples in {s}: {low.to_dict()}")

# ---- leakage evidence: distance from test points to nearest train point ----
def nn_km(a, b):
    tree = BallTree(np.radians(b[["lat", "lon"]].values), metric="haversine")
    d, _ = tree.query(np.radians(a[["lat", "lon"]].values), k=1)
    return d[:, 0] * 6371.0

tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
d_spatial = nn_km(te, tr)
rand = df.sample(frac=1.0, random_state=SEED)
n_te = len(te)
d_random = nn_km(rand.iloc[:n_te], rand.iloc[n_te:])
print("\nnearest-train-neighbour distance (km), percentiles 10/50/90:")
print("  spatial split:", np.percentile(d_spatial, [10, 50, 90]).round(2))
print("  random split :", np.percentile(d_random, [10, 50, 90]).round(2))

# ---- save ----
Path("data/splits").mkdir(parents=True, exist_ok=True)
out = f"data/splits/spatial_seed{SEED}_cell{CELL:g}.csv"
df[["fname", "classification_label", "dataset", "lat", "lon", "cell", "split"]].to_csv(out, index=False)
print("saved", out)