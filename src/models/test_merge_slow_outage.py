"""
Side experiment:
Merge SLOW_INTERNET + INTERNET_OUTAGE into NETWORK_CONNECTIVITY.

This does NOT modify the original dataset or production classifier.

Compares:
1. Original 8-class classification
2. Experimental 7-class classification

Model:
Character TF-IDF + Logistic Regression

Evaluation:
Stratified 5-fold cross-validation
Accuracy
Macro F1
Weighted F1
Per-class F1
Confusion matrix
"""

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "data/processed/merged_training_dataset_cleaned.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "intent"

N_SPLITS = 5
RANDOM_STATE = 42

MERGED_LABEL = "NETWORK_CONNECTIVITY"

MERGE_CLASSES = {
    "SLOW_INTERNET",
    "INTERNET_OUTAGE",
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("SIDE EXPERIMENT: MERGE SLOW_INTERNET + INTERNET_OUTAGE")
print("=" * 80)

df = pd.read_csv(CSV_PATH)

print(f"\nOriginal dataset: {df.shape}")


# ============================================================
# REMOVE PSEUDO-LABELS IF YOUR DATASET HAS THEM
# ============================================================

if "label_source" in df.columns:

    before = len(df)

    df = df[
        df["label_source"] != "tfidf_pseudo_label"
    ].copy()

    print(
        f"Excluded pseudo-labeled rows: "
        f"{before - len(df)}"
    )


# ============================================================
# REMOVE RARE CLASSES
# Same logic as your current experiment
# ============================================================

class_counts = df[LABEL_COLUMN].value_counts()

rare_classes = class_counts[
    class_counts < 10
].index.tolist()

rare_df = df[
    df[LABEL_COLUMN].isin(rare_classes)
].copy()

main_df = df[
    ~df[LABEL_COLUMN].isin(rare_classes)
].copy()

print("\nRare classes kept out of evaluation:")

for label in rare_classes:
    print(
        f"  {label}: "
        f"{class_counts[label]} examples"
    )

print(
    f"\nMain evaluation dataset: "
    f"{main_df.shape}"
)


# ============================================================
# ORIGINAL LABELS
# ============================================================

X = main_df[TEXT_COLUMN].fillna("")

y_original = main_df[LABEL_COLUMN].copy()


# ============================================================
# MERGED LABELS
#
# IMPORTANT:
# We create a COPY.
# Original y_original remains untouched.
# ============================================================

y_merged = y_original.copy()

y_merged[
    y_merged.isin(MERGE_CLASSES)
] = MERGED_LABEL


# ============================================================
# SHOW CLASS DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("ORIGINAL CLASS DISTRIBUTION")
print("=" * 80)

print(
    y_original.value_counts()
)


print("\n" + "=" * 80)
print("MERGED CLASS DISTRIBUTION")
print("=" * 80)

print(
    y_merged.value_counts()
)


# ============================================================
# MODEL
# Same model as current experiment
# ============================================================

pipeline = Pipeline([
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
            max_iter=2000
        ),
    ),
])


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


scoring = [
    "accuracy",
    "f1_macro",
    "f1_weighted",
]


# ============================================================
# ORIGINAL 8-CLASS MODEL
# ============================================================

print("\n" + "=" * 80)
print("ORIGINAL 8-CLASS MODEL")
print("=" * 80)


original_results = cross_validate(
    pipeline,
    X,
    y_original,
    cv=cv,
    scoring=scoring,
)


original_accuracy = (
    original_results["test_accuracy"].mean()
)

original_macro = (
    original_results["test_f1_macro"].mean()
)

original_weighted = (
    original_results["test_f1_weighted"].mean()
)


print(
    f"\nAccuracy:     {original_accuracy:.4f}"
)

print(
    f"Macro F1:     {original_macro:.4f}"
)

print(
    f"Weighted F1:  {original_weighted:.4f}"
)


# ============================================================
# MERGED 7-CLASS MODEL
# ============================================================

print("\n" + "=" * 80)
print("MERGED 7-CLASS MODEL")
print("=" * 80)

merged_results = cross_validate(
    pipeline,
    X,
    y_merged,
    cv=cv,
    scoring=scoring,
)


merged_accuracy = (
    merged_results["test_accuracy"].mean()
)

merged_macro = (
    merged_results["test_f1_macro"].mean()
)

merged_weighted = (
    merged_results["test_f1_weighted"].mean()
)


print(
    f"\nAccuracy:     {merged_accuracy:.4f}"
)

print(
    f"Macro F1:     {merged_macro:.4f}"
)

print(
    f"Weighted F1:  {merged_weighted:.4f}"
)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("8-CLASS VS 7-CLASS COMPARISON")
print("=" * 80)


print(
    f"\nAccuracy:"
    f"\n  Original: {original_accuracy:.4f}"
    f"\n  Merged:   {merged_accuracy:.4f}"
    f"\n  Change:   {merged_accuracy - original_accuracy:+.4f}"
)


print(
    f"\nMacro F1:"
    f"\n  Original: {original_macro:.4f}"
    f"\n  Merged:   {merged_macro:.4f}"
    f"\n  Change:   {merged_macro - original_macro:+.4f}"
)


print(
    f"\nWeighted F1:"
    f"\n  Original: {original_weighted:.4f}"
    f"\n  Merged:   {merged_weighted:.4f}"
    f"\n  Change:   {merged_weighted - original_weighted:+.4f}"
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if merged_macro > original_macro:

    print(
        "\nMacro F1 increased after merging."
    )

    print(
        "The merged class is easier for the model "
        "to distinguish than the original two classes."
    )

else:

    print(
        "\nMacro F1 did not increase after merging."
    )

    print(
        "The merge did not provide an overall "
        "macro-F1 improvement in this experiment."
    )


print(
    "\nIMPORTANT:"
)

print(
    "This experiment does NOT change the original dataset."
)

print(
    "It only tests an alternative label structure."
)

print(
    "\n" + "=" * 80
)
print("EXPERIMENT COMPLETE")
print("=" * 80)