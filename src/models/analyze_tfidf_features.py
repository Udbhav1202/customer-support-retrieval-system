import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = "data/processed/merged_training_dataset_cleaned.csv"


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("TF-IDF FEATURE ANALYSIS")
print("=" * 80)

print("\nDataset:", df.shape)


# ============================================================
# Remove TF-IDF pseudo-labeled data
# ============================================================

df = df[
    df["label_source"] != "tfidf_pseudo_label"
].copy()

print("Human-labeled dataset:", df.shape)


# ============================================================
# Remove rare intents
# ============================================================

intent_counts = df["intent"].value_counts()

valid_intents = intent_counts[
    intent_counts >= 5
].index

df = df[
    df["intent"].isin(valid_intents)
].copy()

print("\nDataset used:", df.shape)


# ============================================================
# Character TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=10000
)

X = vectorizer.fit_transform(df["text"].fillna(""))

features = vectorizer.get_feature_names_out()

print("\nNumber of TF-IDF features:", len(features))


# ============================================================
# Analyze strongest features per intent
# ============================================================

print("\n" + "=" * 80)
print("TOP FEATURES PER INTENT")
print("=" * 80)


for intent in sorted(df["intent"].unique()):

    subset = (df["intent"] == intent).to_numpy()

    class_matrix = X[subset]

    scores = class_matrix.mean(axis=0).A1

    top_indices = scores.argsort()[-20:][::-1]

    print(f"\n{intent}")

    for idx in top_indices:

        print(
            f"  {features[idx]:20s} "
            f"{scores[idx]:.4f}"
        )


# ============================================================
# Word TF-IDF for human-readable features
# ============================================================

print("\n" + "=" * 80)
print("TOP WORD FEATURES PER INTENT")
print("=" * 80)


word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_features=10000
)

X_word = word_vectorizer.fit_transform(
    df["text"].fillna("")
)

word_features = word_vectorizer.get_feature_names_out()


for intent in sorted(df["intent"].unique()):

    subset = (df["intent"] == intent).to_numpy()

    class_matrix = X_word[subset]

    scores = class_matrix.mean(axis=0).A1

    top_indices = scores.argsort()[-15:][::-1]

    print(f"\n{intent}")

    for idx in top_indices:

        print(
            f"  {word_features[idx]:25s} "
            f"{scores[idx]:.4f}"
        )


print("\n" + "=" * 80)
print("FEATURE ANALYSIS COMPLETE")
print("=" * 80)