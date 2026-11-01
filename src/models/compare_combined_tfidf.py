import pandas as pd

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, accuracy_score


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path(
    "data/processed/merged_training_dataset_cleaned.csv"
)

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("WORD + CHARACTER TF-IDF EXPERIMENT")
print("=" * 80)

print(f"\nDataset: {df.shape}")


# Remove pseudo-labeled examples
df = df[
    df["label_source"] != "tfidf_pseudo_label"
].copy()

print(f"Human-labeled dataset: {df.shape}")


# Remove rare intents in the same way as baseline
intent_counts = df["intent"].value_counts()

rare_intents = intent_counts[
    intent_counts < N_SPLITS
].index.tolist()

if rare_intents:
    print("\nExcluded rare intents:")

    for intent in rare_intents:
        print(
            f"  {intent}: {intent_counts[intent]} examples"
        )

    df = df[
        ~df["intent"].isin(rare_intents)
    ].copy()


print(f"\nDataset used: {df.shape}")

print("\nIntent distribution:")
print(df["intent"].value_counts())


# ============================================================
# DATA
# ============================================================

X = df["text"].fillna("").astype(str)
y = df["intent"]


# ============================================================
# COMBINED TF-IDF
# ============================================================

features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
    ),

    (
        "char",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True
        )
    )
])


# ============================================================
# MODEL
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)


pipeline = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([
    ("features", features),
    ("classifier", model)
])


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


scoring = {
    "accuracy": make_scorer(accuracy_score),
    "macro_f1": make_scorer(
        f1_score,
        average="macro"
    ),
    "weighted_f1": make_scorer(
        f1_score,
        average="weighted"
    )
}


print("\n" + "=" * 80)
print("COMBINED WORD + CHARACTER TF-IDF")
print("=" * 80)


results = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)


# ============================================================
# RESULTS
# ============================================================

accuracy_scores = results["test_accuracy"]
macro_f1_scores = results["test_macro_f1"]
weighted_f1_scores = results["test_weighted_f1"]


print(
    "\nAccuracy scores:",
    [round(x, 4) for x in accuracy_scores]
)

print(
    f"Mean accuracy: {accuracy_scores.mean():.4f}"
)

print(
    f"Std accuracy: {accuracy_scores.std():.4f}"
)


print(
    "\nMacro F1 scores:",
    [round(x, 4) for x in macro_f1_scores]
)

print(
    f"Mean macro F1: {macro_f1_scores.mean():.4f}"
)

print(
    f"Std macro F1: {macro_f1_scores.std():.4f}"
)


print(
    "\nWeighted F1 scores:",
    [round(x, 4) for x in weighted_f1_scores]
)

print(
    f"Mean weighted F1: {weighted_f1_scores.mean():.4f}"
)

print(
    f"Std weighted F1: {weighted_f1_scores.std():.4f}"
)


print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)