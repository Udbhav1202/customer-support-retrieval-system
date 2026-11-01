
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

LABELS_PATH = PROCESSED / "hybrid_manual_review_v2.csv"

df = pd.read_csv(LABELS_PATH)

required = {"query_id", "rank", "relevance"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df[list(required)].isna().any().any():
    raise ValueError("Missing query IDs, ranks, or relevance scores.")

if not df["relevance"].isin([0, 1, 2]).all():
    raise ValueError("Relevance scores must be 0, 1, or 2.")

if df.duplicated(["query_id", "rank"]).any():
    raise ValueError("Duplicate query/rank pairs found.")

counts = df.groupby("query_id").size()
if not (counts == 5).all():
    raise ValueError("Each query must have exactly 5 judged results.")

def dcg(scores):
    return sum(
        score / math.log2(rank + 1)
        for rank, score in enumerate(scores, start=1)
    )

precision_scores = []
reciprocal_ranks = []
ndcg_scores = []

for _, group in df.groupby("query_id"):
    group = group.sort_values("rank")
    scores = group["relevance"].tolist()

    # Precision@5: only score 2 counts as directly relevant.
    precision_scores.append(
        sum(score == 2 for score in scores) / 5
    )

    # MRR: reciprocal rank of the first directly relevant result.
    first_relevant_rank = next(
        (
            rank
            for rank, score in enumerate(scores, start=1)
            if score == 2
        ),
        None,
    )
    reciprocal_ranks.append(
        1 / first_relevant_rank if first_relevant_rank else 0
    )

    # NDCG@5: use raw relevance scores 0, 1, and 2.
    actual_dcg = dcg(scores)
    ideal_dcg = dcg(sorted(scores, reverse=True))
    ndcg_scores.append(
        actual_dcg / ideal_dcg if ideal_dcg else 0
    )

print("Queries evaluated:", df["query_id"].nunique())
print(f"Precision@5: {sum(precision_scores) / len(precision_scores):.4f}")
print(f"MRR:         {sum(reciprocal_ranks) / len(reciprocal_ranks):.4f}")
print(f"NDCG@5:      {sum(ndcg_scores) / len(ndcg_scores):.4f}")
print("\nRelevance distribution:")
print(df["relevance"].value_counts().sort_index().to_string())
