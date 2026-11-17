import json
from pathlib import Path
import pandas as pd

with open("data/raw/labels.geojson") as f:
    gj = json.load(f)
df = pd.DataFrame([ft["properties"] for ft in gj["features"]])

files = {p.name for p in Path("features/arrays").glob("*.h5")}
df["fname"] = df["index"].astype(str) + "_" + df["dataset"].astype(str) + ".h5"
df["has_feat"] = df["fname"].isin(files)

print("labels with a feature file:", df["has_feat"].sum(), "of", len(df))
print("feature files matching no label:", len(files - set(df["fname"])))
print("\nhas_feat by is_test:")
print(df.groupby("is_test")["has_feat"].agg(["sum", "count"]))

multi = df[df["classification_label"].notna()]
print("\nmulticlass rows:", len(multi), "| with features:", multi["has_feat"].sum())
print(multi[multi["has_feat"]]["classification_label"].value_counts())
print("\nsource datasets in usable multiclass set:",
      multi[multi["has_feat"]]["dataset"].nunique())