
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

PATH = PROCESSED / "hybrid_manual_review_v2.csv"
df = pd.read_csv(PATH)

required = {"query_id", "query", "expected_intent", "rank", "relevance"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df[list(required)].isna().any().any():
    raise ValueError("Missing required values.")

if not df["relevance"].isin([0, 1, 2]).all():
    raise ValueError("Relevance must be 0, 1, or 2.")

# One row per query, with the number of directly relevant results.
per_query = (
    df.assign(directly_relevant=(df["relevance"] == 2).astype(int))
    .groupby(["query_id", "query", "expected_intent"], as_index=False)
    .agg(
        relevant_at_5=("directly_relevant", "sum"),
        weakly_relevant=("relevance", lambda s: (s == 1).sum()),
        irrelevant=("relevance", lambda s: (s == 0).sum()),
    )
)

per_query["precision_at_5"] = per_query["relevant_at_5"] / 5

by_intent = (
    per_query.groupby("expected_intent", as_index=False)
    .agg(
        queries=("query_id", "count"),
        mean_precision_at_5=("precision_at_5", "mean"),
        directly_relevant=("relevant_at_5", "sum"),
        weakly_relevant=("weakly_relevant", "sum"),
        irrelevant=("irrelevant", "sum"),
    )
    .sort_values("mean_precision_at_5")
)

print("Precision@5 by intent:")
print(by_intent.to_string(index=False))

print("\nQueries with the lowest Precision@5:")
print(
    per_query.sort_values(
        ["relevant_at_5", "query_id"]
    ).head(10).to_string(index=False)
)
