
from pathlib import Path
import pandas as pd

LABELS_PATH = Path("data/processed/bm25_relevance_labels.csv")
REVIEWED_PATH = Path("data/processed/bm25_results_to_review.csv")
OUTPUT_PATH = Path("data/processed/bm25_relevance_labels_final.csv")

labels = pd.read_csv(LABELS_PATH)
reviewed = pd.read_csv(REVIEWED_PATH)

# Normalize IDs so matching is reliable.
labels["customer_tweet_id"] = labels["customer_tweet_id"].astype(str)
reviewed["customer_tweet_id"] = reviewed["customer_tweet_id"].astype(str)

keys = ["query_id", "customer_tweet_id"]

if reviewed.duplicated(keys).any():
    raise ValueError("Duplicate query/document pairs found in reviewed CSV.")

reviewed["relevance"] = pd.to_numeric(
    reviewed["relevance"], errors="coerce"
)

if reviewed["relevance"].isna().any():
    raise ValueError("Some reviewed rows have missing or invalid relevance scores.")

if not reviewed["relevance"].isin([0, 1, 2]).all():
    raise ValueError("Relevance scores must be 0, 1, or 2.")

# Match every reviewed row to exactly one BM25 result.
label_index = labels.set_index(keys).index
review_index = pd.MultiIndex.from_frame(reviewed[keys])

missing_keys = ~review_index.isin(label_index)
if missing_keys.any():
    raise ValueError("Some reviewed rows do not match BM25 results.")

for _, row in reviewed.iterrows():
    mask = (
        (labels["query_id"] == row["query_id"])
        & (labels["customer_tweet_id"] == row["customer_tweet_id"])
    )

    if mask.sum() != 1:
        raise ValueError("A reviewed row did not match exactly one result.")

    if labels.loc[mask, "relevance"].notna().any():
        raise ValueError("A reviewed row would overwrite an existing judgment.")

    labels.loc[mask, "relevance"] = int(row["relevance"])

# Validate the final dataset.
if len(labels) != 150:
    raise ValueError(f"Expected 150 results, found {len(labels)}.")

if labels["relevance"].isna().any():
    raise ValueError(
        f"{labels['relevance'].isna().sum()} results still need labels."
    )

labels["relevance"] = labels["relevance"].astype(int)
labels.to_csv(OUTPUT_PATH, index=False)

print(f"Total results: {len(labels)}")
print(f"Unlabeled results: {labels['relevance'].isna().sum()}")
print("\nRelevance distribution:")
print(labels["relevance"].value_counts().sort_index())
print(f"\nSaved final labels to: {OUTPUT_PATH}")
