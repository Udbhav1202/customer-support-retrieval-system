import pandas as pd

from src.retrieval.tfidf_retriever import TfidfRetriever


DATA_PATH = "data/processed/verizon_retrieval_cases_clean.csv"


df = pd.read_csv(DATA_PATH)

retriever = TfidfRetriever(df)


queries = [
    "my internet is very slow",
    "my wifi is not working",
    "I need help with my bill",
    "my router is not connecting"
]


for query in queries:
    print("=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    results = retriever.retrieve(query, top_k=3)

    for i, row in results.iterrows():
        print(f"\nSimilarity: {row['similarity']:.4f}")
        print("Customer:", row["customer_message"])
        print("Support:", row["support_response"])