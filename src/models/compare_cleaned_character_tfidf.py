import pandas as pd
import re

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

DATA_PATH = Path(
    "data/processed/merged_training_dataset_cleaned.csv"
)

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("RAW vs CLEANED CHARACTER TF-IDF")
print("=" * 80)

print(f"\nDataset: {df.shape}")


# ============================================================
# Keep human-labeled data
# ============================================================

df = df[
    df["label_source"] != "tfidf_pseudo_label"
].copy()


# ============================================================
# Remove rare intents
# Same logic as baseline
# ============================================================

intent_counts = df["intent"].value_counts()

rare_intents = intent_counts[
    intent_counts < N_SPLITS
].index.tolist()

if rare_intents:
    print("\nExcluded rare intents:")

    for intent in rare_intents:
        print(
            f"  {intent}: "
            f"{intent_counts[intent]} examples"
        )

    df = df[
        ~df["intent"].isin(rare_intents)
    ].copy()


print(f"\nDataset used: {df.shape}")


# ============================================================
# Text cleaning
# ============================================================

def clean_text(text):

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove Twitter-style @mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["clean_text"] = (
    df["text"]
    .fillna("")
    .apply(clean_text)
)


# ============================================================
# Prepare X / y
# ============================================================

X_raw = df["text"].fillna("")
X_clean = df["clean_text"]

y = df["intent"]
groups = df["conversation_id"]

# ============================================================
# Character TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=1,
)


model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
)

pipeline = Pipeline([
    ("tfidf", vectorizer),
    ("classifier", model)
])


# ============================================================
# Cross-validation
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = [
    "accuracy",
    "f1_macro",
    "f1_weighted"
]


# ============================================================
# RAW TEXT
# ============================================================

print("\n" + "=" * 80)
print("RAW TEXT + CHARACTER TF-IDF")
print("=" * 80)


raw_results = cross_validate(
    pipeline,
    X_raw,
    y,
    groups=groups,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
)


print(
    "\nAccuracy:",
    round(
        raw_results["test_accuracy"].mean(),
        4
    )
)

print(
    "Macro F1:",
    round(
        raw_results["test_f1_macro"].mean(),
        4
    )
)

print(
    "Weighted F1:",
    round(
        raw_results["test_f1_weighted"].mean(),
        4
    )
)


# ============================================================
# CLEANED TEXT
# ============================================================

print("\n" + "=" * 80)
print("CLEANED TEXT + CHARACTER TF-IDF")
print("=" * 80)

clean_results = cross_validate(
    pipeline,
    X_clean,
    y,
    groups=groups,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
)


print(
    "\nAccuracy:",
    round(
        clean_results["test_accuracy"].mean(),
        4
    )
)

print(
    "Macro F1:",
    round(
        clean_results["test_f1_macro"].mean(),
        4
    )
)

print(
    "Weighted F1:",
    round(
        clean_results["test_f1_weighted"].mean(),
        4
    )
)


# ============================================================
# COMPARISON
# ============================================================

raw_accuracy = raw_results[
    "test_accuracy"
].mean()

clean_accuracy = clean_results[
    "test_accuracy"
].mean()

raw_macro = raw_results[
    "test_f1_macro"
].mean()

clean_macro = clean_results[
    "test_f1_macro"
].mean()

raw_weighted = raw_results[
    "test_f1_weighted"
].mean()

clean_weighted = clean_results[
    "test_f1_weighted"
].mean()


print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(
    f"\nAccuracy:"
    f"\n  Raw:     {raw_accuracy:.4f}"
    f"\n  Cleaned: {clean_accuracy:.4f}"
    f"\n  Change:  {clean_accuracy - raw_accuracy:+.4f}"
)

print(
    f"\nMacro F1:"
    f"\n  Raw:     {raw_macro:.4f}"
    f"\n  Cleaned: {clean_macro:.4f}"
    f"\n  Change:  {clean_macro - raw_macro:+.4f}"
)

print(
    f"\nWeighted F1:"
    f"\n  Raw:     {raw_weighted:.4f}"
    f"\n  Cleaned: {clean_weighted:.4f}"
    f"\n  Change:  {clean_weighted - raw_weighted:+.4f}"
)

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)