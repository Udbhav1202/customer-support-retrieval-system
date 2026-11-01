
from pathlib import Path
import pandas as pd

old = pd.read_csv("data/processed/tfidf_relevance_labels.csv")
bm25 = pd.read_csv("data/processed/bm25_retrieval_results.csv")

old["customer_tweet_id"] = old["customer_tweet_id"].astype(str)
bm25["customer_tweet_id"] = bm25["customer_tweet_id"].astype(str)

# Count matching documents regardless of query.
tweet_matches = bm25.merge(
    old[["customer_tweet_id", "query_id", "query", "relevance"]],
    on="customer_tweet_id",
    how="inner",
    suffixes=("_bm25", "_tfidf"),
)

# Count matches for the same query and document.
same_query_matches = bm25.merge(
    old[["query_id", "customer_tweet_id", "relevance"]],
    on=["query_id", "customer_tweet_id"],
    how="inner",
)

print("BM25 results:", len(bm25))
print("Matching rows regardless of query:", len(tweet_matches))
print("Matching rows for same query + document:", len(same_query_matches))
print("Unique matching documents:", tweet_matches["customer_tweet_id"].nunique())

print("\nExamples of documents found under different queries:")
different = tweet_matches[
    tweet_matches["query_id_bm25"] != tweet_matches["query_id_tfidf"]
]
print(
    different[
        ["query_id_bm25", "query_id_tfidf", "customer_tweet_id", "relevance"]
    ].head(10).to_string(index=False)
)
