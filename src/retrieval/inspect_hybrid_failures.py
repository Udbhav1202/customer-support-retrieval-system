from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

results = pd.read_csv(
    PROCESSED / "hybrid_retrieval_results.csv"
)

labels = pd.read_csv(
    PROCESSED / "hybrid_relevance_labels_final.csv"
)

failure_ids = [13, 20, 29]

merged = results.merge(
    labels[["query_id", "rank", "relevance"]],
    on=["query_id", "rank"],
    how="left",
    validate="one_to_one",
)

for query_id in failure_ids:
    group = merged[merged["query_id"] == query_id].sort_values("rank")

    print(f"\n{'=' * 80}")
    print(f"QUERY ID: {query_id}")

    for _, row in group.iterrows():
        print(f"\nRank: {row['rank']} | Relevance: {row['relevance']}")
        print(f"Customer: {row['customer_message']}")
        print(f"Support:  {row['support_response']}")
