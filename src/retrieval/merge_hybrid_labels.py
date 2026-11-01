
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

LABELS_PATH = PROCESSED / "hybrid_relevance_labels.csv"
REVIEW_PATH = PROCESSED / "hybrid_results_to_review.csv"
OUTPUT_PATH = PROCESSED / "hybrid_relevance_labels_final.csv"

labels = pd.read_csv(LABELS_PATH)
review = pd.read_csv(REVIEW_PATH)

keys = ["query", "customer_tweet_id"]

def normalize_query(series):
    return series.astype(str).str.strip().str.lower()

labels["_query_key"] = normalize_query(labels["query"])
review["_query_key"] = normalize_query(review["query"])

review["relevance"] = pd.to_numeric(
    review["relevance"], errors="coerce"
)

if review["relevance"].isna().any():
    raise ValueError("Some reviewed rows have missing relevance scores.")

if not review["relevance"].isin([0, 1, 2]).all():
    raise ValueError("Relevance scores must be 0, 1, or 2.")

review_scores = review[
    ["_query_key", "customer_tweet_id", "relevance"]
].copy()

if review_scores.duplicated(
    ["_query_key", "customer_tweet_id"]
).any():
    raise ValueError("Duplicate cases found in the review file.")

labels = labels.merge(
    review_scores.rename(columns={"relevance": "_review_relevance"}),
    on=["_query_key", "customer_tweet_id"],
    how="left",
    validate="one_to_one",
)

labels["relevance"] = labels["_review_relevance"].combine_first(
    labels["relevance"]
)

labels = labels.drop(
    columns=["_query_key", "_review_relevance"]
)

if labels["relevance"].isna().any():
    raise ValueError(
        f"{labels['relevance'].isna().sum()} relevance scores are still missing."
    )

labels["relevance"] = labels["relevance"].astype(int)
labels.to_csv(OUTPUT_PATH, index=False)

print("Total hybrid results:", len(labels))
print("Missing relevance scores:", labels["relevance"].isna().sum())
print("Relevance distribution:")
print(labels["relevance"].value_counts().sort_index())
print("Saved final labels:", OUTPUT_PATH)
