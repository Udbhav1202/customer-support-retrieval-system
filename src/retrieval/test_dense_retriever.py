
import pandas as pd

from src.retrieval.dense_retriever import DenseRetriever

DATA_PATH = "data/processed/verizon_retrieval_cases_clean.csv"

df = pd.read_csv(DATA_PATH)

retriever = DenseRetriever(df)

query = "I cannot connect to my wireless network"

results = retriever.retrieve(query, top_k=5)

print(f"\nQuery: {query}\n")

for rank, (_, row) in enumerate(results.iterrows(), start=1):
    print(f"Rank: {rank}")
    print(f"Similarity: {row['similarity']:.4f}")
    print(f"Customer: {row['customer_message']}")
    print(f"Support: {row['support_response']}")
    print("-" * 80)
