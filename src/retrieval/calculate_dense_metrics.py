
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

labels_path = PROCESSED / "dense_relevance_labels_final.csv"
df = pd.read_csv(labels_path)

required = {"query", "rank", "relevance"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df[list(required)].isna().any().any():
    raise ValueError("Missing query, rank, or relevance values.")

if not df["relevance"].isin([0, 1, 2]).all():
    raise ValueError("Relevance scores must be 0, 1, or 2.")

df["rank"] = pd.to_numeric(df["rank"], errors="raise")
df["relevance"] = pd.to_numeric(df["relevance"], errors="raise")

if df.duplicated(["query", "rank"]).any():
    raise ValueError("Duplicate query/rank pairs found.")

precision_scores = []
reciprocal_ranks = []
ndcg_scores = []

for query, group in df.groupby("query"):
    group = group.sort_values("rank")
    top_k = group.head(5)

    # Precision@5: proportion of top-five results rated fully relevant.
    precision_scores.append((top_k["relevance"] == 2).sum() / 5)

    # MRR: reciprocal rank of the first fully relevant result.
    relevant_ranks = top_k.loc[
        top_k["relevance"] == 2, "rank"
    ]
    reciprocal_ranks.append(
        1 / relevant_ranks.iloc[0] if not relevant_ranks.empty else 0
    )

    # NDCG@5: relevance gains 0, 1, 2 with logarithmic rank discount.
    gains = top_k["relevance"].to_numpy(dtype=float)
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains / discounts)

    ideal_gains = np.sort(group["relevance"].to_numpy(dtype=float))[::-1][:5]
    ideal_discounts = np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = np.sum(ideal_gains / ideal_discounts)

    ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

print("Dense Retrieval Evaluation")
print("--------------------------")
print("Queries:", df["query"].nunique())
print("Labeled results:", len(df))
print(f"Precision@5: {np.mean(precision_scores):.4f}")
print(f"MRR:         {np.mean(reciprocal_ranks):.4f}")
print(f"NDCG@5:      {np.mean(ndcg_scores):.4f}")
