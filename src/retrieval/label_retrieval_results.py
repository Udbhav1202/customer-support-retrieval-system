
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/tfidf_retrieval_results.csv")
OUTPUT_PATH = Path("data/processed/tfidf_relevance_labels.csv")

results = pd.read_csv(INPUT_PATH)

if OUTPUT_PATH.exists():
    labels = pd.read_csv(OUTPUT_PATH)
    completed = set(
        zip(labels["query_id"], labels["rank"])
    )
else:
    labels = pd.DataFrame()
    completed = set()

print("Relevance scale:")
print("0 = Irrelevant")
print("1 = Related, but not useful enough")
print("2 = Useful historical support case")
print("q = Save and quit")

for _, row in results.iterrows():
    key = (row["query_id"], row["rank"])

    if key in completed:
        continue

    print("\n" + "=" * 70)
    print(f"Query: {row['query']}")
    print(f"Expected intent: {row['expected_intent']}")
    print(f"Rank: {row['rank']}")
    print(f"Similarity: {row['similarity']:.4f}")

    print(f"\nHistorical customer: {row['customer_message']}")
    print(f"Historical support: {row['support_response']}")

    while True:
        rating = input("\nRelevance (0/1/2/q): ").strip().lower()
        if rating in {"0", "1", "2", "q"}:
            break
        print("Enter 0, 1, 2, or q.")

    if rating == "q":
        break

    labeled_row = row.to_dict()
    labeled_row["relevance"] = int(rating)

    labels = pd.concat(
        [labels, pd.DataFrame([labeled_row])],
        ignore_index=True
    )

    labels.to_csv(OUTPUT_PATH, index=False)
    completed.add(key)

print(f"\nSaved {len(labels)} labels to {OUTPUT_PATH}")