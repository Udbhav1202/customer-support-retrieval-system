
import pandas as pd

LABELS_PATH = "data/processed/tfidf_relevance_labels.csv"
QUERIES_PATH = "data/processed/retrieval_eval_queries.csv"

labels = pd.read_csv(LABELS_PATH)
queries = pd.read_csv(QUERIES_PATH)

# The weakest query IDs from the evaluation output.
weak_ids = [14, 11, 28, 19, 13, 18, 10, 3, 25, 4]

for query_id in weak_ids:
    group = labels[labels["query_id"] == query_id].sort_values("rank")

    if group.empty:
        continue

    query_text = group["query"].iloc[0]
    print("\n" + "=" * 90)
    print(f"Query ID: {query_id} | Query: {query_text}")
    print(f"Expected intent: {group['expected_intent'].iloc[0]}")

    for _, row in group.iterrows():
        print(f"\nRank {row['rank']} | Relevance: {row['relevance']}")
        print(f"Customer: {row['customer_message']}")
        print(f"Support:  {row['support_response']}")