import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["customer_message"]
        )

    def retrieve(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        top_indices = similarities.argsort()[::-1][:top_k]

        results = self.df.iloc[top_indices].copy()
        results["similarity"] = similarities[top_indices]

        return results[
            [
                "conversation_id",
                "customer_tweet_id",
                "support_tweet_id",
                "customer_message",
                "support_response",
                "similarity"
            ]
        ]