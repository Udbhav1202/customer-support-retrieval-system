
import math
from pathlib import Path

import pandas as pd

LABELS_PATH = Path("data/processed/bm25_relevance_labels_final.csv")

df = pd.read_csv(LABELS_PATH)

required_columns = {"query_id", "rank", "relevance"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

if df["relevance"].isna().any():
    raise ValueError("Some BM25 results have no relevance labels.")

if df.duplicated(["query_id", "rank"]).any():
    raise ValueError("Duplicate query/rank pairs found.")

if not df["relevance"].isin([0, 1, 2]).all():
    raise ValueError("Relevance scores must be 0, 1, or 2.")

df["is_relevant"] = (df["relevance"] == 2).astype(int)


def reciprocal_rank(group):
    relevant_ranks = group.loc[
        group["is_relevant"] == 1, "rank"
    ]
    return 1 / relevant_ranks.min() if not relevant_ranks.empty else 0.0


def ndcg_at_5(group):
    group = group.sort_values("rank")
    gains = group["relevance"].tolist()[:5]

    discounts = [
        1 / math.log2(rank + 1)
        for rank in range(1, len(gains) + 1)
    ]

    dcg = sum(
        gain * discount
        for gain, discount in zip(gains, discounts)
    )

    ideal_gains = sorted(gains, reverse=True)
    ideal_dcg = sum(
        gain * discount
        for gain, discount in zip(ideal_gains, discounts)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0


precision = df.groupby("query_id")["is_relevant"].mean()
mrr = df.groupby("query_id").apply(reciprocal_rank)
ndcg = df.groupby("query_id").apply(ndcg_at_5)

print(f"Queries evaluated: {df['query_id'].nunique()}")
print(f"Results scored: {len(df)}")
print(f"Precision@5: {precision.mean():.4f}")
print(f"MRR:          {mrr.mean():.4f}")
print(f"NDCG@5:       {ndcg.mean():.4f}")
