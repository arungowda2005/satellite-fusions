
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import BallTree

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CELL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

N_STARTS = 12
N_ITER = 30000
MIN_CLASS_COUNT = 30

FRAC = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

SPLITS = list(FRAC.keys())


# --------------------------------------------------
# LOAD USABLE MULTICLASS SAMPLES
# --------------------------------------------------
with open("data/raw/labels.geojson", encoding="utf-8") as f:
    gj = json.load(f)

df = pd.DataFrame([feature["properties"] for feature in gj["features"]])

files = {
    p.name for p in Path("features/arrays").glob("*.h5")
}

df["fname"] = (
    df["index"].astype(str)
    + "_"
    + df["dataset"].astype(str)
    + ".h5"
)

df = df[
    df["classification_label"].notna()
    & df["fname"].isin(files)
].copy()

df["lat"] = pd.to_numeric(df["lat"])
df["lon"] = pd.to_numeric(df["lon"])

print("Usable multiclass samples:", len(df))
print("is_test=True samples:", int(df["is_test"].sum()))


# --------------------------------------------------
# CLASS AND GEOGRAPHIC CELL COUNTS
# --------------------------------------------------
classes = sorted(df["classification_label"].unique())

df["cell"] = (
    np.floor(df["lat"] / CELL).astype(int).astype(str)
    + "_"
    + np.floor(df["lon"] / CELL).astype(int).astype(str)
)

cc = pd.crosstab(
    df["cell"],
    df["classification_label"]
).reindex(columns=classes, fill_value=0)

cell_names = cc.index.to_numpy()
matrix = cc.to_numpy(dtype=float)

n_cells, n_classes = matrix.shape
total_classes = matrix.sum(axis=0)
total_samples = total_classes.sum()

print(f"\nGrid size: {CELL} degrees")
print("Number of geographic cells:", n_cells)
print(
    "Largest cell (%):",
    round(matrix.sum(axis=1).max() / total_samples * 100, 2)
)

print("\nTotal samples per class:")
print(pd.Series(total_classes, index=classes))


# --------------------------------------------------
# OBJECTIVE FUNCTION
# --------------------------------------------------
fractions = np.array([FRAC[s] for s in SPLITS])

target_classes = (
    fractions[:, None] * total_classes[None, :]
)

target_sizes = fractions * total_samples

def calculate_score(counts, sizes):
    # Normalized class-distribution error
    class_error = np.mean(
        ((counts - target_classes) /
         np.maximum(target_classes, 1)) ** 2
    )

    # Overall split-size error
    size_error = np.mean(
        ((sizes - target_sizes) /
         np.maximum(target_sizes, 1)) ** 2
    )

    # Penalize rare classes below the minimum in val/test.
    rare_penalty = 0.0

    for split_index in [1, 2]:
        deficits = np.maximum(
            MIN_CLASS_COUNT - counts[split_index],
            0
        )

        rare_penalty += np.sum(
            (deficits / MIN_CLASS_COUNT) ** 2
        )

    # The weights prioritize class balance and rare-class coverage.
    return (
        4.0 * class_error
        + 2.0 * size_error
        + 2.0 * rare_penalty
    )


# --------------------------------------------------
# INITIAL GREEDY ASSIGNMENT
# --------------------------------------------------
def make_initial_assignment(rng):
    assignment = np.full(n_cells, -1, dtype=int)

    counts = np.zeros((3, n_classes))
    sizes = np.zeros(3)

    cell_sizes = matrix.sum(axis=1)

    order = np.argsort(
        -(cell_sizes * rng.uniform(0.8, 1.2, n_cells))
    )

    for cell_index in order:
        vector = matrix[cell_index]
        size = cell_sizes[cell_index]

        best_split = None
        best_score = float("inf")

        for split_index in range(3):
            trial_counts = counts.copy()
            trial_sizes = sizes.copy()

            trial_counts[split_index] += vector
            trial_sizes[split_index] += size

            score = calculate_score(
                trial_counts,
                trial_sizes
            )

            if score < best_score:
                best_score = score
                best_split = split_index

        assignment[cell_index] = best_split
        counts[best_split] += vector
        sizes[best_split] += size

    return assignment


# --------------------------------------------------
# LOCAL SEARCH
# --------------------------------------------------
def optimize_assignment(initial, rng):
    assignment = initial.copy()

    counts = np.zeros((3, n_classes))
    sizes = np.zeros(3)

    for i, split_index in enumerate(assignment):
        counts[split_index] += matrix[i]
        sizes[split_index] += matrix[i].sum()

    current_score = calculate_score(counts, sizes)

    best_assignment = assignment.copy()
    best_score = current_score

    for step in range(N_ITER):
        cell_index = rng.integers(n_cells)
        old_split = assignment[cell_index]
        new_split = rng.integers(3)

        if new_split == old_split:
            continue

        # Do not empty a geographic split.
        if np.sum(assignment == old_split) <= 1:
            continue

        vector = matrix[cell_index]
        size = vector.sum()

        trial_counts = counts.copy()
        trial_sizes = sizes.copy()

        trial_counts[old_split] -= vector
        trial_counts[new_split] += vector

        trial_sizes[old_split] -= size
        trial_sizes[new_split] += size

        trial_score = calculate_score(
            trial_counts,
            trial_sizes
        )

        # Simulated annealing: occasionally accept a worse move
        # early in the search to escape local minima.
        progress = step / N_ITER
        temperature = max(0.001, 0.05 * (1 - progress))

        accept = (
            trial_score < current_score
            or rng.random() < np.exp(
                (current_score - trial_score) / temperature
            )
        )

        if accept:
            assignment[cell_index] = new_split
            counts = trial_counts
            sizes = trial_sizes
            current_score = trial_score

            if current_score < best_score:
                best_score = current_score
                best_assignment = assignment.copy()

    return best_assignment, best_score


# --------------------------------------------------
# MULTI-START OPTIMIZATION
# --------------------------------------------------
print("\nRunning multi-start spatial optimization...")

best_assignment = None
best_score = float("inf")

for start in range(N_STARTS):
    rng = np.random.default_rng(SEED + start)

    initial = make_initial_assignment(rng)

    candidate, score = optimize_assignment(
        initial,
        rng
    )

    print(
        f"Candidate {start + 1}/{N_STARTS}: "
        f"score = {score:.6f}"
    )

    if score < best_score:
        best_score = score
        best_assignment = candidate.copy()

print("\nBest objective score:", round(best_score, 6))


# --------------------------------------------------
# ASSIGN SPLITS TO SAMPLES
# --------------------------------------------------
split_names = np.array(SPLITS)

cell_split = {
    cell_names[i]: split_names[best_assignment[i]]
    for i in range(n_cells)
}

df["split"] = df["cell"].map(cell_split)


# --------------------------------------------------
# REPORT SPLIT SIZES AND CLASS COUNTS
# --------------------------------------------------
print("\nSplit proportions:")
print(df["split"].value_counts(normalize=True).reindex(SPLITS).round(4))

tab = pd.crosstab(
    df["classification_label"],
    df["split"]
).reindex(columns=SPLITS, fill_value=0)

print("\nClass counts per split:")
print(tab)

print("\nClasses below minimum count:")

for split in ["val", "test"]:
    low = tab[split][tab[split] < MIN_CLASS_COUNT]

    print(f"{split}: {low.to_dict()}")


# --------------------------------------------------
# GEOGRAPHIC CELL OVERLAP CHECK
# --------------------------------------------------
print("\nGeographic cell overlap check:")

cell_sets = {
    split: set(df.loc[df["split"] == split, "cell"])
    for split in SPLITS
}

for i, first in enumerate(SPLITS):
    for second in SPLITS[i + 1:]:
        overlap = cell_sets[first] & cell_sets[second]

        print(
            f"{first} vs {second}: "
            f"{len(overlap)} shared cells"
        )

assert not (cell_sets["train"] & cell_sets["val"])
assert not (cell_sets["train"] & cell_sets["test"])
assert not (cell_sets["val"] & cell_sets["test"])

print("PASS: no geographic cell is shared between splits.")


# --------------------------------------------------
# NEAREST TRAIN NEIGHBOUR DISTANCES
# --------------------------------------------------
def nn_km(a, b):
    tree = BallTree(
        np.radians(b[["lat", "lon"]].values),
        metric="haversine"
    )

    distances, _ = tree.query(
        np.radians(a[["lat", "lon"]].values),
        k=1
    )

    return distances[:, 0] * 6371.0


train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

spatial_distances = nn_km(test, train)

random_df = df.sample(
    frac=1.0,
    random_state=SEED
)

random_distances = nn_km(
    random_df.iloc[:len(test)],
    random_df.iloc[len(test):]
)

print("\nNearest-train-neighbour distance (km)")
print("Percentiles: 10 / 50 / 90")

print(
    "Spatial:",
    np.percentile(
        spatial_distances,
        [10, 50, 90]
    ).round(2)
)

print(
    "Random:",
    np.percentile(
        random_distances,
        [10, 50, 90]
    ).round(2)
)


# --------------------------------------------------
# SAVE EXPERIMENTAL RESULT
# --------------------------------------------------
output_dir = Path("data/splits")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = (
    output_dir
    / f"optimized_spatial_seed{SEED}_cell{CELL:g}.csv"
)

df[
    [
        "fname",
        "classification_label",
        "dataset",
        "lat",
        "lon",
        "cell",
        "split"
    ]
].to_csv(output_file, index=False)

print("\nSaved:", output_file)
