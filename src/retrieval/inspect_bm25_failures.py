
from pathlib import Path

import pandas as pd

LABELS_PATH = Path(
    "data/processed/bm25_relevance_labels_final.csv"
)

df = pd.read_csv(LABELS_PATH)

df["is_relevant"] = (df["relevance"] == 2).astype(int)

precision = df.groupby(["query_id", "query", "expected_intent"])[
    "is_relevant"
].mean()

print("BM25: 10 queries with the lowest Precision@5\n")

print(
    precision.sort_values()
    .head(10)
    .to_string()
)
