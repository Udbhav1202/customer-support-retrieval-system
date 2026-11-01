
"""
Side experiment:
Compare 8-class, 7-class, and 6-class intent taxonomies
using the same 5-fold conversation-aware cross-validation.

IMPORTANT:
- This script does NOT modify the original dataset.
- This script does NOT modify the production classifier.
- It only compares alternative label structures.

Taxonomies tested:

1. ORIGINAL 8 classes

2. NETWORK MERGE
   SLOW_INTERNET + INTERNET_OUTAGE
   -> NETWORK_CONNECTIVITY

3. SUPPORT MERGE
   CUSTOMER_SUPPORT_ACCESS + CUSTOMER_SUPPORT_SERVICE_FEEDBACK
   -> CUSTOMER_SUPPORT

4. BOTH MERGES
   Both of the above.

Usage:

    python src/models/compare_taxonomies_cv.py
"""

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "data/processed/merged_training_dataset_cleaned.csv"

TEXT_COL = "text"
LABEL_COL = "intent"
GROUP_COL = "conversation_id"
SOURCE_COL = "label_source"

PSEUDO_LABEL_SOURCE = "tfidf_pseudo_label"

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    df = pd.read_csv(CSV_PATH)

    print("=" * 80)
    print("TAXONOMY COMPARISON - 5 FOLD CV")
    print("=" * 80)

    print(f"\nOriginal dataset: {df.shape}")

    # Remove pseudo-labelled examples from evaluation
    if SOURCE_COL in df.columns:

        before = len(df)

        df = df[
            df[SOURCE_COL] != PSEUDO_LABEL_SOURCE
        ].copy()

        removed = before - len(df)

        print(
            f"Excluded pseudo-labeled rows: {removed}"
        )

    # Keep rare classes out of evaluation.
    # They are not included in the main evaluation dataset.
    counts = df[LABEL_COL].value_counts()

    rare_labels = counts[counts < 10].index.tolist()

    if rare_labels:

        print("\nRare classes kept out of evaluation:")

        for label in rare_labels:
            print(
                f"  {label}: "
                f"{counts[label]} examples"
            )

        df = df[
            ~df[LABEL_COL].isin(rare_labels)
        ].copy()

    df = df.reset_index(drop=True)

    print(
        f"\nEvaluation dataset: {df.shape}"
    )

    print("\nClass distribution:")

    print(
        df[LABEL_COL].value_counts()
    )

    return df


# ============================================================
# MODEL
# ============================================================

def build_model():

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        sublinear_tf=True,
    )

    classifier = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )
    )

    return vectorizer, classifier


# ============================================================
# TAXONOMIES
# ============================================================

def apply_taxonomy(df, taxonomy):

    result = df.copy()

    if taxonomy == "ORIGINAL_8":

        # No changes
        pass

    elif taxonomy == "NETWORK_MERGE":

        result[LABEL_COL] = result[
            LABEL_COL
        ].replace(
            {
                "SLOW_INTERNET": "NETWORK_CONNECTIVITY",
                "INTERNET_OUTAGE": "NETWORK_CONNECTIVITY",
            }
        )

    elif taxonomy == "SUPPORT_MERGE":

        result[LABEL_COL] = result[
            LABEL_COL
        ].replace(
            {
                "CUSTOMER_SUPPORT_ACCESS":
                    "CUSTOMER_SUPPORT",

                "CUSTOMER_SUPPORT_SERVICE_FEEDBACK":
                    "CUSTOMER_SUPPORT",
            }
        )

    elif taxonomy == "BOTH_MERGES":

        result[LABEL_COL] = result[
            LABEL_COL
        ].replace(
            {
                "SLOW_INTERNET":
                    "NETWORK_CONNECTIVITY",

                "INTERNET_OUTAGE":
                    "NETWORK_CONNECTIVITY",

                "CUSTOMER_SUPPORT_ACCESS":
                    "CUSTOMER_SUPPORT",

                "CUSTOMER_SUPPORT_SERVICE_FEEDBACK":
                    "CUSTOMER_SUPPORT",
            }
        )

    else:

        raise ValueError(
            f"Unknown taxonomy: {taxonomy}"
        )

    return result


# ============================================================
# CROSS VALIDATION
# ============================================================

def evaluate_taxonomy(df, taxonomy):

    data = apply_taxonomy(
        df,
        taxonomy
    )

    X = data[TEXT_COL].values
    y = data[LABEL_COL].values
    groups = data[GROUP_COL].values

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_accuracy = []
    fold_macro_f1 = []
    fold_weighted_f1 = []

    all_true = []
    all_pred = []

    print("\n")
    print("=" * 80)
    print(f"TAXONOMY: {taxonomy}")
    print("=" * 80)

    print("\nClass distribution:")

    print(
        data[LABEL_COL].value_counts()
    )

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1,
    ):

        print(
            f"\nRunning fold "
            f"{fold}/{N_SPLITS}..."
        )

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        vectorizer, classifier = build_model()

        X_train_tfidf = vectorizer.fit_transform(
            X_train
        )

        X_test_tfidf = vectorizer.transform(
            X_test
        )

        classifier.fit(
            X_train_tfidf,
            y_train
        )

        predictions = classifier.predict(
            X_test_tfidf
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        fold_accuracy.append(
            accuracy
        )

        fold_macro_f1.append(
            macro_f1
        )

        fold_weighted_f1.append(
            weighted_f1
        )

        all_true.extend(y_test)
        all_pred.extend(predictions)

        print(
            f"  Accuracy:     {accuracy:.4f}"
        )

        print(
            f"  Macro F1:     {macro_f1:.4f}"
        )

        print(
            f"  Weighted F1:  {weighted_f1:.4f}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("-" * 80)
    print(f"{taxonomy} - CV SUMMARY")
    print("-" * 80)

    print(
        f"Accuracy:    "
        f"{np.mean(fold_accuracy):.4f} "
        f"+/- "
        f"{np.std(fold_accuracy):.4f}"
    )

    print(
        f"Macro F1:    "
        f"{np.mean(fold_macro_f1):.4f} "
        f"+/- "
        f"{np.std(fold_macro_f1):.4f}"
    )

    print(
        f"Weighted F1: "
        f"{np.mean(fold_weighted_f1):.4f} "
        f"+/- "
        f"{np.std(fold_weighted_f1):.4f}"
    )

    # ========================================================
    # PER-CLASS F1
    # ========================================================

    labels = sorted(
        data[LABEL_COL].unique()
    )

    per_class_f1 = f1_score(
        all_true,
        all_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    print("\nPer-class F1 across OOF predictions:")

    for label, score in zip(
        labels,
        per_class_f1
    ):

        print(
            f"  {label:<40} "
            f"{score:.4f}"
        )

    return {
        "taxonomy": taxonomy,

        "accuracy_mean":
            np.mean(fold_accuracy),

        "accuracy_std":
            np.std(fold_accuracy),

        "macro_f1_mean":
            np.mean(fold_macro_f1),

        "macro_f1_std":
            np.std(fold_macro_f1),

        "weighted_f1_mean":
            np.mean(fold_weighted_f1),

        "weighted_f1_std":
            np.std(fold_weighted_f1),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    taxonomies = [
        "ORIGINAL_8",
        "NETWORK_MERGE",
        "SUPPORT_MERGE",
        "BOTH_MERGES",
    ]

    results = []

    for taxonomy in taxonomies:

        result = evaluate_taxonomy(
            df,
            taxonomy
        )

        results.append(result)

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 80)
    print("FINAL 5-FOLD CV COMPARISON")
    print("=" * 80)

    print()

    for _, row in results_df.iterrows():

        print(
            f"{row['taxonomy']:<20} "
            f"Accuracy: "
            f"{row['accuracy_mean']:.4f} "
            f"+/- "
            f"{row['accuracy_std']:.4f} | "
            f"Macro F1: "
            f"{row['macro_f1_mean']:.4f} "
            f"+/- "
            f"{row['macro_f1_std']:.4f} | "
            f"Weighted F1: "
            f"{row['weighted_f1_mean']:.4f} "
            f"+/- "
            f"{row['weighted_f1_std']:.4f}"
        )

    # ========================================================
    # CHANGE VS ORIGINAL
    # ========================================================

    baseline = results_df[
        results_df["taxonomy"] == "ORIGINAL_8"
    ].iloc[0]

    print("\n")
    print("=" * 80)
    print("CHANGE VS ORIGINAL 8-CLASS MODEL")
    print("=" * 80)

    for _, row in results_df.iterrows():

        if row["taxonomy"] == "ORIGINAL_8":
            continue

        macro_change = (
            row["macro_f1_mean"]
            - baseline["macro_f1_mean"]
        )

        accuracy_change = (
            row["accuracy_mean"]
            - baseline["accuracy_mean"]
        )

        weighted_change = (
            row["weighted_f1_mean"]
            - baseline["weighted_f1_mean"]
        )

        print(
            f"\n{row['taxonomy']}"
        )

        print(
            f"  Accuracy change:    "
            f"{accuracy_change:+.4f}"
        )

        print(
            f"  Macro F1 change:    "
            f"{macro_change:+.4f}"
        )

        print(
            f"  Weighted F1 change: "
            f"{weighted_change:+.4f}"
        )

    print("\n")
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        "\nOriginal dataset was NOT modified."
    )


if __name__ == "__main__":
    main()
