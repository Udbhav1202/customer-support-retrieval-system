
from pathlib import Path

import pandas as pd

LABELS_PATH = Path("data/processed/tfidf_relevance_labels.csv")

df = pd.read_csv(LABELS_PATH)

required_columns = {"query_id", "rank", "relevance"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Binary relevance: only score 2 counts as fully relevant.
df["is_relevant"] = (df["relevance"] == 2).astype(int)

# Precision@5 for each query.
precision_at_5 = (
    df.groupby("query_id")["is_relevant"].sum() / 5
)

# Reciprocal rank of the first fully relevant result.
def reciprocal_rank(group):
    relevant = group.loc[group["is_relevant"] == 1, "rank"]
    if relevant.empty:
        return 0.0
    return 1.0 / relevant.min()

mrr = df.groupby("query_id").apply(reciprocal_rank)

# NDCG@5 using the original graded relevance scores (0, 1, 2).
def ndcg_at_5(group):
    group = group.sort_values("rank")
    gains = group["relevance"].to_numpy()
    discounts = [
        1 / __import__("math").log2(rank + 1)
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

ndcg = df.groupby("query_id").apply(ndcg_at_5)

print(f"Queries evaluated: {df['query_id'].nunique()}")
print(f"Results labeled: {len(df)}")
print(f"Precision@5: {precision_at_5.mean():.4f}")
print(f"MRR:          {mrr.mean():.4f}")
print(f"NDCG@5:       {ndcg.mean():.4f}")

print("\nLowest Precision@5 queries:")
print(precision_at_5.sort_values().head(10).to_string())

print("\nLowest NDCG@5 queries:")
print(ndcg.sort_values().head(10).to_string())