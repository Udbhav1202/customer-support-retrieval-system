import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.base import clone


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/merged_training_dataset_cleaned.csv"

N_SPLITS = 5
RANDOM_STATE = 42

OOF_OUTPUT_PATH = (
    "data/processed/character_tfidf_oof_predictions.csv"
)


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("5-FOLD CROSS-VALIDATION BASELINE")
print("=" * 80)

print(f"\nOriginal dataset: {df.shape}")


# ============================================================
# Keep human-labeled examples only
# ============================================================

df = df[
    df["label_source"] != "tfidf_pseudo_label"
].copy()

print(
    f"Human-labeled dataset: {df.shape}"
)


# ============================================================
# Remove rows without required fields
# ============================================================

df = df.dropna(
    subset=[
        "conversation_id",
        "text",
        "intent",
    ]
).copy()


# ============================================================
# Remove classes with fewer than 5 examples
#
# These classes are too small for reliable 5-fold CV.
# ============================================================

counts = df["intent"].value_counts()

valid_intents = counts[
    counts >= N_SPLITS
].index

excluded_intents = counts[
    counts < N_SPLITS
]


if len(excluded_intents) > 0:

    print("\nExcluded rare intents:")

    for intent, count in excluded_intents.items():

        print(
            f"  {intent}: {count} examples"
        )


df = df[
    df["intent"].isin(valid_intents)
].copy()


# ============================================================
# Dataset information
# ============================================================

X = df["text"].fillna("")
y = df["intent"]
groups = df["conversation_id"]


print("\nDataset used for cross-validation:")
print(df.shape)

print("\nIntent distribution:")
print(y.value_counts())

print(
    "\nUnique conversations:",
    groups.nunique()
)


# ============================================================
# Cross-validation
#
# StratifiedGroupKFold:
#
# 1. Keeps class distribution approximately balanced.
# 2. Keeps the same conversation out of multiple folds.
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# Experiments
# ============================================================

experiments = [

    (
        "Word TF-IDF (1,2)",

        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        ),
    ),

    (
        "Word TF-IDF no stopwords",

        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
        ),
    ),

    (
        "Character TF-IDF",

        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1,
        ),
    ),
]


# ============================================================
# Run experiments
# ============================================================

for name, vectorizer in experiments:

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    pipeline = Pipeline([

        (
            "tfidf",
            vectorizer,
        ),

        (
            "classifier",

            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
        ),
    ])


    scores = cross_validate(
        pipeline,
        X,
        y,
        groups=groups,
        cv=cv,

        scoring=[
            "accuracy",
            "f1_macro",
            "f1_weighted",
        ],

        return_train_score=False,
    )


    accuracy_scores = scores[
        "test_accuracy"
    ]

    macro_f1_scores = scores[
        "test_f1_macro"
    ]

    weighted_f1_scores = scores[
        "test_f1_weighted"
    ]


    # ========================================================
    # Accuracy
    # ========================================================

    print(
        "\nAccuracy scores:",
        [
            round(float(x), 4)
            for x in accuracy_scores
        ],
    )

    print(
        "Mean accuracy:",
        round(
            accuracy_scores.mean(),
            4,
        ),
    )

    print(
        "Std accuracy:",
        round(
            accuracy_scores.std(),
            4,
        ),
    )


    # ========================================================
    # Macro F1
    # ========================================================

    print(
        "\nMacro F1 scores:",
        [
            round(float(x), 4)
            for x in macro_f1_scores
        ],
    )

    print(
        "Mean macro F1:",
        round(
            macro_f1_scores.mean(),
            4,
        ),
    )

    print(
        "Std macro F1:",
        round(
            macro_f1_scores.std(),
            4,
        ),
    )


    # ========================================================
    # Weighted F1
    # ========================================================

    print(
        "\nWeighted F1 scores:",
        [
            round(float(x), 4)
            for x in weighted_f1_scores
        ],
    )

    print(
        "Mean weighted F1:",
        round(
            weighted_f1_scores.mean(),
            4,
        ),
    )

    print(
        "Std weighted F1:",
        round(
            weighted_f1_scores.std(),
            4,
        ),
    )


# ============================================================
# Detailed Character TF-IDF evaluation
#
# Character TF-IDF is currently our strongest baseline.
#
# We collect predictions from every validation fold so we can
# inspect per-intent performance and the confusion matrix.
# ============================================================

print("\n" + "=" * 80)
print("DETAILED CHARACTER TF-IDF EVALUATION")
print("=" * 80)


character_pipeline = Pipeline([

    (
        "tfidf",

        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1,
        ),
    ),

    (
        "classifier",

        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
    ),
])


all_predictions = []
all_actual = []
all_conversation_ids = []
all_texts = []


for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups),
    start=1,
):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]


    model = clone(
        character_pipeline
    )


    model.fit(
        X_train,
        y_train,
    )


    predictions = model.predict(
        X_val
    )

    all_conversation_ids.extend(
        df.iloc[val_idx]["conversation_id"].tolist()
    )

    all_texts.extend(
        X_val.tolist()
    )


    all_predictions.extend(
        predictions
    )

    all_actual.extend(
        y_val.tolist()
    )


    print(
        f"\nFold {fold} accuracy:",
        round(
            accuracy_score(
                y_val,
                predictions,
            ),
            4,
        ),
    )


# ============================================================
# Combined out-of-fold results
# ============================================================

print("\n" + "=" * 80)
print("COMBINED OUT-OF-FOLD RESULTS")
print("=" * 80)


overall_accuracy = accuracy_score(
    all_actual,
    all_predictions,
)


overall_macro_f1 = f1_score(
    all_actual,
    all_predictions,
    average="macro",
)


overall_weighted_f1 = f1_score(
    all_actual,
    all_predictions,
    average="weighted",
)


print(
    "\nAccuracy:",
    round(
        overall_accuracy,
        4,
    ),
)


print(
    "Macro F1:",
    round(
        overall_macro_f1,
        4,
    ),
)


print(
    "Weighted F1:",
    round(
        overall_weighted_f1,
        4,
    ),
)


# ============================================================
# Per-class classification report
# ============================================================

print("\n" + "=" * 80)
print("PER-INTENT PERFORMANCE")
print("=" * 80)


report = classification_report(
    all_actual,
    all_predictions,
    zero_division=0,
)


print(report)


# ============================================================
# Confusion matrix
# ============================================================

labels = sorted(
    y.unique()
)


cm = confusion_matrix(
    all_actual,
    all_predictions,
    labels=labels,
)


cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels,
)


print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(
    cm_df.to_string()
)

# ============================================================
# Save out-of-fold predictions
# ============================================================

oof_df = pd.DataFrame({
    "conversation_id": all_conversation_ids,
    "text": all_texts,
    "actual_intent": all_actual,
    "predicted_intent": all_predictions,
})


oof_df.to_csv(
    OOF_OUTPUT_PATH,
    index=False,
)


print("\n" + "=" * 80)
print("OUT-OF-FOLD PREDICTIONS SAVED")
print("=" * 80)

print(
    f"\nRows saved: {len(oof_df)}"
)

print(
    f"Saved: {OOF_OUTPUT_PATH}"
)
