
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

scaffold_path = PROCESSED / "dense_relevance_labels.csv"
review_path = PROCESSED / "dense_results_to_review.csv"
output_path = PROCESSED / "dense_relevance_labels_final.csv"

labels = pd.read_csv(scaffold_path)
review = pd.read_csv(review_path)

required = {"query", "customer_tweet_id", "relevance"}
for name, df in [("Scaffold", labels), ("Review", review)]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

def normalize_query(series):
    return series.astype(str).str.strip().str.lower()

# Build stable matching keys.
labels["_query_key"] = normalize_query(labels["query"])
review["_query_key"] = normalize_query(review["query"])

labels["_tweet_key"] = labels["customer_tweet_id"].astype(str).str.strip()
review["_tweet_key"] = review["customer_tweet_id"].astype(str).str.strip()

keys = ["_query_key", "_tweet_key"]

# Each query/customer-case combination must be unique.
if labels.duplicated(keys).any():
    raise ValueError("Duplicate query + customer_tweet_id keys in scaffold.")

if review.duplicated(keys).any():
    raise ValueError("Duplicate query + customer_tweet_id keys in review.")

# Confirm the review rows correspond exactly to unlabeled scaffold rows.
scaffold_keys = set(map(tuple, labels[keys].to_numpy()))
review_keys = set(map(tuple, review[keys].to_numpy()))

unmatched = review_keys - scaffold_keys
if unmatched:
    raise ValueError(
        f"{len(unmatched)} reviewed rows do not match the scaffold."
    )

review_lookup = review.set_index(keys)["relevance"]
matched_scores = pd.MultiIndex.from_frame(labels[keys]).map(review_lookup)

needs_score = labels["relevance"].isna()
if (needs_score & pd.isna(matched_scores)).any():
    raise ValueError("Some unlabeled scaffold rows have no reviewed score.")

# Preserve existing judgments and fill only missing scores.
labels.loc[needs_score, "relevance"] = matched_scores[needs_score]

labels["relevance"] = pd.to_numeric(labels["relevance"], errors="raise")

if labels["relevance"].isna().any():
    raise ValueError("Missing relevance scores remain.")

if not labels["relevance"].isin([0, 1, 2]).all():
    raise ValueError("Unexpected relevance scores found.")

labels = labels.drop(columns=["_query_key", "_tweet_key"])
labels.to_csv(output_path, index=False)

print("Merge successful!")
print("Total labeled results:", len(labels))
print("Missing relevance scores:", labels["relevance"].isna().sum())
print("Score distribution:")
print(labels["relevance"].value_counts().sort_index())
print("Saved to:", output_path)
