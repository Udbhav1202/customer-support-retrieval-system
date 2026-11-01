
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

HYBRID_PATH = PROCESSED / "hybrid_retrieval_results_v2.csv"

LABEL_FILES = [
    PROCESSED / "tfidf_relevance_labels.csv",
    PROCESSED / "bm25_relevance_labels_final.csv",
    PROCESSED / "dense_relevance_labels_final.csv",
]

OUTPUT_PATH = PROCESSED / "hybrid_relevance_labels_v2.csv"
REVIEW_PATH = PROCESSED / "hybrid_results_to_review_v2.csv"


def normalize_query(series):
    return series.astype(str).str.strip().str.lower()


hybrid = pd.read_csv(HYBRID_PATH)
existing = pd.concat(
    [pd.read_csv(path) for path in LABEL_FILES],
    ignore_index=True,
)

required = {"query", "customer_tweet_id"}
label_required = {"query", "customer_tweet_id", "relevance"}

missing_hybrid = required - set(hybrid.columns)
if missing_hybrid:
    raise ValueError(
        f"Hybrid results missing columns: {sorted(missing_hybrid)}"
    )

missing_labels = label_required - set(existing.columns)
if missing_labels:
    raise ValueError(
        f"Existing labels missing columns: {sorted(missing_labels)}"
    )

hybrid["_query_key"] = normalize_query(hybrid["query"])
existing["_query_key"] = normalize_query(existing["query"])

existing = existing[
    ["_query_key", "customer_tweet_id", "relevance"]
].copy()

conflicts = existing.groupby(
    ["_query_key", "customer_tweet_id"]
)["relevance"].nunique()

if (conflicts > 1).any():
    raise ValueError(
        "Conflicting relevance judgments found."
    )

existing = existing.drop_duplicates(
    ["_query_key", "customer_tweet_id"]
).rename(
    columns={"relevance": "_existing_relevance"}
)

merged = hybrid.merge(
    existing,
    on=["_query_key", "customer_tweet_id"],
    how="left",
    validate="many_to_one",
)

merged["relevance"] = merged["_existing_relevance"]

review = merged[merged["relevance"].isna()].copy()
review["relevance"] = ""

merged = merged.drop(
    columns=["_query_key", "_existing_relevance"]
)
review = review.drop(
    columns=["_query_key", "_existing_relevance"]
)

merged.to_csv(OUTPUT_PATH, index=False)
review.to_csv(REVIEW_PATH, index=False)

print("Hybrid results:", len(hybrid))
print("Existing judgments reused:", merged["relevance"].notna().sum())
print("Results needing review:", len(review))
print("Saved labels:", OUTPUT_PATH)
print("Saved review file:", REVIEW_PATH)
