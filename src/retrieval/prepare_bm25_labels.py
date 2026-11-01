
from pathlib import Path
import pandas as pd

TFIDF_LABELS = Path("data/processed/tfidf_relevance_labels.csv")
BM25_RESULTS = Path("data/processed/bm25_retrieval_results.csv")
OUTPUT = Path("data/processed/bm25_relevance_labels.csv")

old = pd.read_csv(TFIDF_LABELS)
bm25 = pd.read_csv(BM25_RESULTS)

# Normalize text for matching query identity.
old["query_key"] = old["query"].astype(str).str.strip().str.lower()
bm25["query_key"] = bm25["query"].astype(str).str.strip().str.lower()

old["customer_tweet_id"] = old["customer_tweet_id"].astype(str)
bm25["customer_tweet_id"] = bm25["customer_tweet_id"].astype(str)

# Reuse a judgment only for the same query text and same document.
judgments = old[
    ["query_key", "customer_tweet_id", "relevance"]
].drop_duplicates(["query_key", "customer_tweet_id"])

labels = bm25.merge(
    judgments,
    on=["query_key", "customer_tweet_id"],
    how="left",
    validate="many_to_one",
)

labels = labels.drop(columns=["query_key"])
labels.to_csv(OUTPUT, index=False)

reused = int(labels["relevance"].notna().sum())
needs_review = int(labels["relevance"].isna().sum())

print(f"BM25 results: {len(labels)}")
print(f"Existing judgments reused: {reused}")
print(f"Results still needing review: {needs_review}")
print(f"Saved to: {OUTPUT}")
