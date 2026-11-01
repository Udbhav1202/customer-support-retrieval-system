
from pathlib import Path

import pandas as pd

DENSE_PATH = Path("data/processed/dense_retrieval_results.csv")
TFIDF_LABELS_PATH = Path("data/processed/tfidf_relevance_labels.csv")
BM25_LABELS_PATH = Path("data/processed/bm25_relevance_labels_final.csv")

OUTPUT_PATH = Path("data/processed/dense_relevance_labels.csv")
REVIEW_PATH = Path("data/processed/dense_results_to_review.csv")


def normalize_query(series):
    return series.astype(str).str.strip().str.lower()


dense = pd.read_csv(DENSE_PATH)

label_files = [
    pd.read_csv(TFIDF_LABELS_PATH),
    pd.read_csv(BM25_LABELS_PATH),
]

existing = pd.concat(label_files, ignore_index=True)

# Keep only the columns needed to identify an existing judgment.
existing = existing[
    ["query", "customer_tweet_id", "relevance"]
].copy()

dense["normalized_query"] = normalize_query(dense["query"])
existing["normalized_query"] = normalize_query(existing["query"])

# The same query and same customer case must match exactly.
existing = existing.drop_duplicates(
    ["normalized_query", "customer_tweet_id"]
)

existing = existing.rename(
    columns={"relevance": "existing_relevance"}
)

merged = dense.merge(
    existing[
        ["normalized_query", "customer_tweet_id", "existing_relevance"]
    ],
    on=["normalized_query", "customer_tweet_id"],
    how="left",
    validate="many_to_one",
)

merged["relevance"] = merged["existing_relevance"]

review_columns = [
    column
    for column in dense.columns
    if column not in ["normalized_query"]
]

to_review = merged[merged["relevance"].isna()][review_columns].copy()

to_review["relevance"] = ""

merged.drop(
    columns=["normalized_query", "existing_relevance"],
    inplace=True,
)

merged.to_csv(OUTPUT_PATH, index=False)
to_review.to_csv(REVIEW_PATH, index=False)

print(f"Dense results: {len(dense)}")
print(f"Existing judgments reused: {merged['relevance'].notna().sum()}")
print(f"Results needing review: {len(to_review)}")
print(f"Saved labels to: {OUTPUT_PATH}")
print(f"Saved review file to: {REVIEW_PATH}")
