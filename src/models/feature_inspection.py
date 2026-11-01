"""
Feature and confusion inspection for the Character TF-IDF +
Logistic Regression intent classifier.

Purpose:
1. Inspect the character n-grams associated with each intent.
2. Compare feature overlap between confused intent pairs.
3. Generate honest out-of-fold predictions using conversation-aware
   StratifiedGroupKFold.
4. Show incorrectly classified examples ranked by confidence margin.

This script DOES NOT modify the dataset.

Project configuration:
- Text column: text
- Label column: intent
- Conversation column: conversation_id
- Label source column: label_source
- Character TF-IDF: analyzer="char", ngram_range=(3,5), min_df=1
- Classifier: LogisticRegression
- Cross-validation: StratifiedGroupKFold
"""

import argparse

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# Load data
# ============================================================

def load_data(
    csv_path,
    source_col=None,
    exclude_sources=None,
):
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("FEATURE AND CONFUSION INSPECTION")
    print("=" * 80)

    print(f"\nDataset loaded: {df.shape}")

    if source_col and exclude_sources:

        before = len(df)

        df = df[
            ~df[source_col].isin(exclude_sources)
        ].reset_index(drop=True)

        removed = before - len(df)

        print(
            f"Excluded {removed} rows "
            f"with source in {exclude_sources}"
        )

    return df


# ============================================================
# Remove rare classes
# ============================================================

def split_rare_classes(
    df,
    label_col,
    min_count,
):

    counts = df[label_col].value_counts()

    rare_labels = counts[
        counts < min_count
    ].index.tolist()

    rare_df = df[
        df[label_col].isin(rare_labels)
    ].reset_index(drop=True)

    main_df = df[
        ~df[label_col].isin(rare_labels)
    ].reset_index(drop=True)

    return (
        main_df,
        rare_df,
        rare_labels,
    )


# ============================================================
# Build Character TF-IDF
# ============================================================

def build_vectorizer():

    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
    )


# ============================================================
# Build classifier
# ============================================================

def build_classifier():

    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )


# ============================================================
# Train model
# ============================================================

def train_model(
    texts,
    labels,
):

    vectorizer = build_vectorizer()

    classifier = build_classifier()

    X = vectorizer.fit_transform(texts)

    classifier.fit(
        X,
        labels,
    )

    return (
        vectorizer,
        classifier,
    )


# ============================================================
# Top features per class
# ============================================================

def top_features_per_class(
    vectorizer,
    classifier,
    top_n,
):

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    class_features = {}

    for class_index, class_name in enumerate(
        classifier.classes_
    ):

        coefficients = classifier.coef_[class_index]

        top_indices = np.argsort(
            coefficients
        )[::-1][:top_n]

        features = []

        for index in top_indices:

            features.append(
                (
                    feature_names[index],
                    round(
                        coefficients[index],
                        4,
                    ),
                )
            )

        class_features[class_name] = features

    return class_features


# ============================================================
# Print top features
# ============================================================

def print_top_features(
    class_features,
    top_n,
):

    print("\n")
    print("=" * 80)
    print("TOP CHARACTER TF-IDF FEATURES PER INTENT")
    print("=" * 80)

    for intent, features in class_features.items():

        print(f"\n{intent}")

        for feature, coefficient in features[:top_n]:

            print(
                f"  {feature!r:<15}"
                f" coefficient={coefficient:.4f}"
            )


# ============================================================
# Compare feature overlap
# ============================================================

def print_pair_overlap(
    class_features,
    pairs,
    top_n,
):

    print("\n")
    print("=" * 80)
    print("FEATURE OVERLAP BETWEEN CONFUSED INTENTS")
    print("=" * 80)

    for intent_a, intent_b in pairs:

        print(
            f"\n{intent_a}"
            f"  VS  "
            f"{intent_b}"
        )

        if (
            intent_a not in class_features
            or intent_b not in class_features
        ):

            print(
                "  One or both intents were not found."
            )

            continue

        features_a = {
            feature
            for feature, _ in
            class_features[intent_a][:top_n]
        }

        features_b = {
            feature
            for feature, _ in
            class_features[intent_b][:top_n]
        }

        shared = features_a & features_b

        only_a = features_a - features_b

        only_b = features_b - features_a

        print(
            f"\n  Shared features "
            f"({len(shared)}):"
        )

        print(
            "   ",
            sorted(shared)
        )

        print(
            f"\n  Distinctive to {intent_a} "
            f"({len(only_a)}):"
        )

        print(
            "   ",
            sorted(only_a)
        )

        print(
            f"\n  Distinctive to {intent_b} "
            f"({len(only_b)}):"
        )

        print(
            "   ",
            sorted(only_b)
        )


# ============================================================
# Out-of-fold predictions
# ============================================================

def generate_oof_predictions(
    df,
    rare_df,
    text_col,
    label_col,
    conversation_col,
    n_splits,
    random_state,
):

    X_text = (
        df[text_col]
        .fillna("")
        .astype(str)
        .values
    )

    y = df[label_col].values

    groups = df[
        conversation_col
    ].values

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    predictions = np.empty(
        len(df),
        dtype=object,
    )

    margins = np.full(
        len(df),
        np.nan,
    )

    probabilities = np.full(
        len(df),
        np.nan,
    )

    for fold, (
        train_indices,
        test_indices,
    ) in enumerate(
        cv.split(
            X_text,
            y,
            groups,
        ),
        start=1,
    ):

        print(
            f"\nGenerating OOF predictions "
            f"for fold {fold}/{n_splits}..."
        )

        train_texts = list(
            X_text[train_indices]
        )

        train_labels = list(
            y[train_indices]
        )

        # Rare classes are kept in training only.
        if len(rare_df) > 0:

            train_texts.extend(
                rare_df[text_col]
                .fillna("")
                .astype(str)
                .tolist()
            )

            train_labels.extend(
                rare_df[label_col]
                .tolist()
            )

        test_texts = X_text[
            test_indices
        ]

        vectorizer = build_vectorizer()

        classifier = build_classifier()

        X_train = vectorizer.fit_transform(
            train_texts
        )

        X_test = vectorizer.transform(
            test_texts
        )

        classifier.fit(
            X_train,
            train_labels,
        )

        fold_predictions = (
            classifier.predict(X_test)
        )

        fold_probabilities = (
            classifier.predict_proba(X_test)
        )

        sorted_probabilities = np.sort(
            fold_probabilities,
            axis=1,
        )

        fold_margins = (
            sorted_probabilities[:, -1]
            -
            sorted_probabilities[:, -2]
        )

        fold_max_probability = (
            sorted_probabilities[:, -1]
        )

        predictions[
            test_indices
        ] = fold_predictions

        margins[
            test_indices
        ] = fold_margins

        probabilities[
            test_indices
        ] = fold_max_probability

    return (
        predictions,
        margins,
        probabilities,
    )


# ============================================================
# Show confusion examples
# ============================================================

def show_confused_examples(
    df,
    text_col,
    label_col,
    predictions,
    margins,
    probabilities,
    pairs,
    top_k=8,
):

    print("\n")
    print("=" * 80)
    print("OUT-OF-FOLD CONFUSION EXAMPLES")
    print("=" * 80)

    print(
        "\nExamples are sorted by confidence margin."
    )

    print(
        "Low margin = model was uncertain."
    )

    print(
        "High margin = model was confident but wrong."
    )

    y_true = df[
        label_col
    ].values

    for intent_a, intent_b in pairs:

        for actual, predicted in [
            (intent_a, intent_b),
            (intent_b, intent_a),
        ]:

            print("\n" + "-" * 80)

            print(
                f"ACTUAL:    {actual}"
            )

            print(
                f"PREDICTED: {predicted}"
            )

            mask = (
                (y_true == actual)
                &
                (predictions == predicted)
            )

            indices = np.where(mask)[0]

            if len(indices) == 0:

                print(
                    "\nNo examples found."
                )

                continue

            indices = indices[
                np.argsort(
                    margins[indices]
                )
            ]

            for index in indices[:top_k]:

                print(
                    f"\nMargin: "
                    f"{margins[index]:.4f}"
                )

                print(
                    f"Confidence: "
                    f"{probabilities[index]:.4f}"
                )

                print(
                    f"Conversation ID: "
                    f"{df.iloc[index]['conversation_id']}"
                )

                print(
                    f"Text:\n"
                    f"{df.iloc[index][text_col]}"
                )

# ============================================================
# Print overall OOF performance
# ============================================================

def print_oof_summary(
    df,
    label_col,
    predictions,
):

    y_true = df[
        label_col
    ].values

    errors = (
        y_true != predictions
    )

    print("\n")
    print("=" * 80)
    print("OOF CONFUSION SUMMARY")
    print("=" * 80)

    print(
        f"\nTotal examples: "
        f"{len(df)}"
    )

    print(
        f"Correct predictions: "
        f"{(~errors).sum()}"
    )

    print(
        f"Incorrect predictions: "
        f"{errors.sum()}"
    )

    print(
        f"Accuracy: "
        f"{(~errors).mean():.4f}"
    )

    error_df = pd.DataFrame(
        {
            "actual": y_true[errors],
            "predicted": predictions[errors],
        }
    )

    if len(error_df) > 0:

        print(
            "\nMost common confusion pairs:"
        )

        print(
            error_df
            .groupby(
                ["actual", "predicted"]
            )
            .size()
            .sort_values(
                ascending=False
            )
            .head(20)
            .to_string()
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default=(
            "data/processed/"
            "merged_training_dataset_cleaned.csv"
        ),
    )

    parser.add_argument(
        "--text-col",
        default="text",
    )

    parser.add_argument(
        "--label-col",
        default="intent",
    )

    parser.add_argument(
        "--conv-col",
        default="conversation_id",
    )

    parser.add_argument(
        "--source-col",
        default="label_source",
    )

    parser.add_argument(
        "--exclude-sources",
        nargs="*",
        default=["tfidf_pseudo_label"],
    )

    parser.add_argument(
        "--rare-min-count",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--confused-pairs",
        nargs="+",
        default=[
            (
                "CUSTOMER_SUPPORT_ACCESS:"
                "INTERNET_OUTAGE"
            ),

            (
                "CUSTOMER_SUPPORT_ACCESS:"
                "CUSTOMER_SUPPORT_SERVICE_FEEDBACK"
            ),

            (
                "INTERNET_OUTAGE:"
                "WIFI_ROUTER"
            ),

            (
                "SLOW_INTERNET:"
                "INTERNET_OUTAGE"
            ),

            (
                "SWITCHING_RETENTION:"
                "OTHER"
            ),
        ],
    )

    args = parser.parse_args()

    pairs = [
        tuple(
            pair.split(":")
        )
        for pair in args.confused_pairs
    ]

    # ========================================================
    # Load dataset
    # ========================================================

    df = load_data(
        args.csv,
        args.source_col,
        args.exclude_sources,
    )

    # ========================================================
    # Validate required columns
    # ========================================================

    required_columns = {
        args.text_col,
        args.label_col,
        args.conv_col,
    }

    missing = (
        required_columns
        -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # ========================================================
    # Remove rare classes from evaluation
    # ========================================================

    (
        main_df,
        rare_df,
        rare_labels,
    ) = split_rare_classes(
        df,
        args.label_col,
        args.rare_min_count,
    )

    print(
        "\nRare classes kept for training only:"
    )

    if rare_labels:

        for label in rare_labels:

            print(
                f"  {label}: "
                f"{len(rare_df[rare_df[args.label_col] == label])}"
            )

    else:

        print("  None")

    print(
        f"\nMain dataset: "
        f"{len(main_df)} rows"
    )

    print(
        f"Rare training-only dataset: "
        f"{len(rare_df)} rows"
    )

    print(
        f"\nIntent distribution:"
    )

    print(
        main_df[
            args.label_col
        ].value_counts()
    )

    # ========================================================
    # Feature interpretation model
    # ========================================================

    print("\n")
    print("=" * 80)
    print("TRAINING FULL DATASET FOR FEATURE INTERPRETATION")
    print("=" * 80)

    all_texts = (
        main_df[args.text_col]
        .fillna("")
        .astype(str)
        .tolist()
    )

    all_labels = (
        main_df[args.label_col]
        .tolist()
    )

    vectorizer, classifier = train_model(
        all_texts,
        all_labels,
    )

    print(
        f"\nNumber of TF-IDF features: "
        f"{len(vectorizer.get_feature_names_out())}"
    )

    class_features = (
        top_features_per_class(
            vectorizer,
            classifier,
            args.top_n,
        )
    )

    print_top_features(
        class_features,
        args.top_n,
    )

    print_pair_overlap(
        class_features,
        pairs,
        args.top_n,
    )

    # ========================================================
    # OOF predictions
    # ========================================================

    print("\n")
    print("=" * 80)
    print("GENERATING OUT-OF-FOLD PREDICTIONS")
    print("=" * 80)

    (
        oof_predictions,
        oof_margins,
        oof_probabilities,
    ) = generate_oof_predictions(
        main_df,
        rare_df,
        args.text_col,
        args.label_col,
        args.conv_col,
        args.folds,
        args.seed,
    )

    # ========================================================
    # Summary
    # ========================================================

    print_oof_summary(
        main_df,
        args.label_col,
        oof_predictions,
    )

    # ========================================================
    # Confused examples
    # ========================================================

    show_confused_examples(
        main_df,
        args.text_col,
        args.label_col,
        oof_predictions,
        oof_margins,
        oof_probabilities,
        pairs,
        top_k=8,
    )

    # ========================================================
    # Complete
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FEATURE INSPECTION COMPLETE")
    print("=" * 80)

    print(
        "\nNo files were modified."
    )


if __name__ == "__main__":
    main()