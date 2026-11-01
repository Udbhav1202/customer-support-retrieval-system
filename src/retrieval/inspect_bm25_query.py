
import pandas as pd

LABELS_PATH = "data/processed/bm25_relevance_labels_final.csv"

df = pd.read_csv(LABELS_PATH)

queries_to_inspect = [
    "I don't know what is wrong",
    "there is a problem with my payment",
    "I cannot connect to my wireless network",
]

for query in queries_to_inspect:
    results = df[df["query"] == query].sort_values("rank")

    print(f"\n{'=' * 90}")
    print(f"QUERY: {query}")
    print(f"EXPECTED INTENT: {results['expected_intent'].iloc[0]}")
    print("=" * 90)

    for _, row in results.iterrows():
        print(f"\nRank: {row['rank']}")
        print(f"Relevance: {row['relevance']}")
        print(f"Customer: {row['customer_message']}")
        print(f"Support: {row['support_response']}")
