
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

INPUT_PATH = PROCESSED / "hybrid_results_to_review_v2.csv"
OUTPUT_PATH = PROCESSED / "hybrid_manual_review_v2.csv"

df = pd.read_csv(INPUT_PATH)

review = df[
    [
        "query_id",
        "query",
        "expected_intent",
        "rank",
        "customer_tweet_id",
        "customer_message",
        "support_response",
    ]
].copy()

review["relevance"] = ""

review = review.sort_values(
    ["query_id", "rank"]
).reset_index(drop=True)

review.to_csv(OUTPUT_PATH, index=False)

print("Rows to review:", len(review))
print("Queries:", review["query_id"].nunique())
print("Saved review sheet:", OUTPUT_PATH)
print("\nRelevance scale:")
print("0 = Irrelevant")
print("1 = Related but weakly useful")
print("2 = Directly relevant and useful")
