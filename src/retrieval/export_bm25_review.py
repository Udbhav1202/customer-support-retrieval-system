
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/bm25_relevance_labels.csv")
OUTPUT_PATH = Path("data/processed/bm25_results_to_review.csv")

df = pd.read_csv(INPUT_PATH)

review = df[df["relevance"].isna()].copy()

review = review[
    [
        "query_id",
        "query",
        "expected_intent",
        "rank",
        "customer_tweet_id",
        "customer_message",
        "support_response",
    ]
]

review["relevance"] = ""

review.to_csv(OUTPUT_PATH, index=False)

print(f"Rows to review: {len(review)}")
print(f"Saved to: {OUTPUT_PATH}")
print("\nAssign relevance:")
print("0 = Irrelevant")
print("1 = Related, but weak")
print("2 = Relevant and useful historical support case")
