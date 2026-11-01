
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/bm25_retrieval_results.csv")
OUTPUT_PATH = Path("data/processed/bm25_relevance_labels.csv")

results = pd.read_csv(INPUT_PATH)

columns = [
    "query_id",
    "query",
    "expected_intent",
    "rank",
    "customer_tweet_id",
    "customer_message",
    "support_response",
]

labels = results[columns].copy()
labels["relevance"] = pd.NA

labels.to_csv(OUTPUT_PATH, index=False)

print(f"Results to label: {len(labels)}")
print(f"Saved to: {OUTPUT_PATH}")
print("\nRelevance scale:")
print("0 = Irrelevant")
print("1 = Related, but weak")
print("2 = Relevant and useful historical support case")