import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

CSV_PATH = "data/processed/merged_training_dataset_cleaned.csv"

TEXT_COL = "text"
TARGET_COL = "intent"

RANDOM_STATE = 42
N_SPLITS = 5


NETWORK_CLASSES = {
    "SLOW_INTERNET",
    "INTERNET_OUTAGE",
}

SUPPORT_CLASSES = {
    "CUSTOMER_SUPPORT_ACCESS",
    "CUSTOMER_SUPPORT_SERVICE_FEEDBACK",
}

RARE_CLASSES = {
    "FRAUD_SECURITY",
    "MOBILE_PHONE_SERVICE",
    "REFUND",
    "SIM_CARD",
}


def apply_6class_taxonomy(intent):
    if intent in NETWORK_CLASSES:
        return "NETWORK_CONNECTIVITY"

    if intent in SUPPORT_CLASSES:
        return "CUSTOMER_SUPPORT"

    return intent


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_dataset():

    df = pd.read_csv(CSV_PATH)

    print(f"Original dataset: {df.shape}")

    # Remove pseudo-labeled rows if present.
    if "is_pseudo_labeled" in df.columns:
        df = df[df["is_pseudo_labeled"] != True].copy()

    # Remove rare classes.
    df = df[~df[TARGET_COL].isin(RARE_CLASSES)].copy()

    # Create 6-class taxonomy.
    df["six_class_intent"] = df[TARGET_COL].apply(
        apply_6class_taxonomy
    )

    # Remove missing text.
    df = df.dropna(subset=[TEXT_COL]).copy()

    # Remove very short texts.
    #
    # Same rule used in test_clean_data_experiment.py:
    # remove texts with word_count <= 4
    df["word_count"] = (
        df[TEXT_COL]
        .astype(str)
        .str.split()
        .str.len()
    )

    before = len(df)

    df = df[df["word_count"] > 4].copy()

    removed = before - len(df)

    print(f"Removed very-short texts: {removed}")
    print(f"Short-removed dataset: {df.shape}")

    return df


def generate_oof_predictions(df):

    X = df[TEXT_COL].astype(str).values
    y = df["six_class_intent"].values

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_true = []
    all_pred = []
    all_confidence = []

    print_section("GENERATING 5-FOLD OOF PREDICTIONS")

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):

        print(f"Generating fold {fold}/{N_SPLITS}...")

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )

        X_train = vectorizer.fit_transform(X[train_idx])
        X_test = vectorizer.transform(X[test_idx])

        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )

        model.fit(
            X_train,
            y[train_idx],
        )

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        confidence = probabilities.max(axis=1)

        all_true.extend(y[test_idx])
        all_pred.extend(predictions)
        all_confidence.extend(confidence)

    return (
        np.array(all_true),
        np.array(all_pred),
        np.array(all_confidence),
    )


def show_performance(y_true, y_pred):

    print_section("SHORT-REMOVED OOF PERFORMANCE")

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
    )

    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")

    print_section("CLASSIFICATION REPORT")

    print(
        classification_report(
            y_true,
            y_pred,
            digits=2,
        )
    )


def show_confusion_matrix(y_true, y_pred):

    labels = sorted(
        np.unique(
            np.concatenate([y_true, y_pred])
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    print_section("CONFUSION MATRIX")

    print(
        pd.DataFrame(
            cm,
            index=[
                f"Actual: {label}"
                for label in labels
            ],
            columns=[
                f"Pred: {label}"
                for label in labels
            ],
        )
    )

    print_section("MOST COMMON CONFUSION PAIRS")

    pairs = []

    for i, actual in enumerate(labels):

        for j, predicted in enumerate(labels):

            if i == j:
                continue

            count = cm[i, j]

            if count > 0:
                pairs.append(
                    (
                        count,
                        actual,
                        predicted,
                    )
                )

    pairs.sort(
        reverse=True
    )

    for count, actual, predicted in pairs[:15]:

        print(
            f"{actual:<30} -> "
            f"{predicted:<30} : {count}"
        )


def show_removed_examples(original_df):

    print_section("REMOVED VERY-SHORT EXAMPLES")

    df = original_df.copy()

    df["word_count"] = (
        df[TEXT_COL]
        .astype(str)
        .str.split()
        .str.len()
    )

    removed = df[
        df["word_count"] <= 4
    ].copy()

    print(
        f"Total removed examples: {len(removed)}"
    )

    print("\nDistribution by 6-class intent:")

    print(
        removed["six_class_intent"]
        .value_counts()
    )

    print("\nExamples:")

    for _, row in removed.iterrows():

        print("\n" + "-" * 80)

        if "conversation_id" in row:
            print(
                f"Conversation ID: "
                f"{row['conversation_id']}"
            )

        print(
            f"Class: "
            f"{row['six_class_intent']}"
        )

        print(
            f"Original intent: "
            f"{row[TARGET_COL]}"
        )

        print(
            f"Word count: "
            f"{row['word_count']}"
        )

        print(
            f"Text: "
            f"{row[TEXT_COL]}"
        )


def main():

    print_section(
        "SHORT-TEXT REMOVAL CONFUSION INSPECTION"
    )

    # Load original data first so we can inspect
    # exactly which examples are removed.
    original_df = pd.read_csv(CSV_PATH)

    if "is_pseudo_labeled" in original_df.columns:
        original_df = original_df[
            original_df["is_pseudo_labeled"] != True
        ].copy()

    original_df = original_df[
        ~original_df[TARGET_COL].isin(RARE_CLASSES)
    ].copy()

    original_df["six_class_intent"] = (
        original_df[TARGET_COL]
        .apply(apply_6class_taxonomy)
    )

    show_removed_examples(
        original_df
    )

    df = load_dataset()

    print_section("6-CLASS DISTRIBUTION")

    print(
        df["six_class_intent"]
        .value_counts()
    )

    y_true, y_pred, confidence = (
        generate_oof_predictions(df)
    )

    show_performance(
        y_true,
        y_pred,
    )

    show_confusion_matrix(
        y_true,
        y_pred,
    )

    print_section("INSPECTION COMPLETE")

    print(
        "\nNo production dataset was modified."
    )


if __name__ == "__main__":
    main()