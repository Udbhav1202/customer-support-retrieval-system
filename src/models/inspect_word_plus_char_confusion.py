import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold


CSV_PATH = "data/processed/merged_training_dataset_cleaned.csv"

TEXT_COL = "text"
LABEL_COL = "intent"

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


def load_dataset():
    df = pd.read_csv(CSV_PATH)

    if "is_pseudo_labeled" in df.columns:
        df = df[df["is_pseudo_labeled"] != True].copy()

    df = df[~df[LABEL_COL].isin(RARE_CLASSES)].copy()

    df["six_class_intent"] = df[LABEL_COL].apply(
        apply_6class_taxonomy
    )

    df = df.dropna(
        subset=[TEXT_COL, "six_class_intent"]
    ).copy()

    return df


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def train_and_predict(
    X_train_text,
    X_test_text,
    y_train,
    experiment,
):
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
    )

    if experiment == "WORD_BIGRAMS":

        X_train = word_vectorizer.fit_transform(
            X_train_text
        )

        X_test = word_vectorizer.transform(
            X_test_text
        )

    elif experiment == "WORD_PLUS_CHAR":

        X_train_word = word_vectorizer.fit_transform(
            X_train_text
        )

        X_test_word = word_vectorizer.transform(
            X_test_text
        )

        X_train_char = char_vectorizer.fit_transform(
            X_train_text
        )

        X_test_char = char_vectorizer.transform(
            X_test_text
        )

        X_train = hstack(
            [
                X_train_word,
                X_train_char,
            ]
        )

        X_test = hstack(
            [
                X_test_word,
                X_test_char,
            ]
        )

    else:
        raise ValueError(
            f"Unknown experiment: {experiment}"
        )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model.predict(X_test)


def run_oof_experiment(
    df,
    experiment,
    splits,
):
    texts = df[TEXT_COL].astype(str).values
    labels = df["six_class_intent"].values

    predictions = np.empty(
        len(df),
        dtype=object,
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splits.split(texts, labels),
        start=1,
    ):

        print(
            f"  {experiment} - Fold {fold}/5..."
        )

        predictions[test_idx] = train_and_predict(
            texts[train_idx],
            texts[test_idx],
            labels[train_idx],
            experiment,
        )

    return predictions


def show_confusion_matrix(
    y_true,
    y_pred,
    labels,
    title,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual: {label}"
            for label in labels
        ],
        columns=[
            f"Pred: {label}"
            for label in labels
        ],
    )

    print_section(title)

    print(
        matrix_df.to_string()
    )


def show_confusion_pairs(
    y_true,
    y_pred,
    labels,
    title,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    pairs = []

    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):

            if i == j:
                continue

            count = matrix[i, j]

            if count > 0:
                pairs.append(
                    (
                        actual,
                        predicted,
                        count,
                    )
                )

    pairs.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    print_section(title)

    for actual, predicted, count in pairs:
        print(
            f"{actual:30} -> "
            f"{predicted:30} : {count}"
        )


def compare_confusion_changes(
    y_true,
    baseline_pred,
    candidate_pred,
    labels,
):
    baseline_matrix = confusion_matrix(
        y_true,
        baseline_pred,
        labels=labels,
    )

    candidate_matrix = confusion_matrix(
        y_true,
        candidate_pred,
        labels=labels,
    )

    print_section(
        "CONFUSION CHANGE: "
        "WORD_PLUS_CHAR vs WORD_BIGRAMS"
    )

    print(
        "Positive value = fewer errors with "
        "WORD_PLUS_CHAR."
    )

    print(
        "Negative value = more errors with "
        "WORD_PLUS_CHAR."
    )

    print()

    for i, actual in enumerate(labels):

        for j, predicted in enumerate(labels):

            if i == j:
                continue

            baseline = baseline_matrix[i, j]
            candidate = candidate_matrix[i, j]

            change = baseline - candidate

            if change != 0:

                print(
                    f"{actual:30} -> "
                    f"{predicted:30} : "
                    f"{baseline} -> {candidate} "
                    f"(change {change:+d})"
                )


def main():

    print_section(
        "WORD vs WORD+CHAR "
        "CONFUSION INSPECTION"
    )

    df = load_dataset()

    print(
        f"\nEvaluation dataset: {df.shape}"
    )

    print("\n6-class distribution:")

    print(
        df["six_class_intent"]
        .value_counts()
    )

    labels = sorted(
        df["six_class_intent"].unique()
    )

    print_section(
        "GENERATING 5-FOLD OOF PREDICTIONS"
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    baseline_pred = run_oof_experiment(
        df,
        "WORD_BIGRAMS",
        cv,
    )

    candidate_pred = run_oof_experiment(
        df,
        "WORD_PLUS_CHAR",
        cv,
    )

    y_true = df[
        "six_class_intent"
    ].values

    print_section(
        "PERFORMANCE COMPARISON"
    )

    baseline_accuracy = accuracy_score(
        y_true,
        baseline_pred,
    )

    baseline_macro = f1_score(
        y_true,
        baseline_pred,
        average="macro",
    )

    baseline_weighted = f1_score(
        y_true,
        baseline_pred,
        average="weighted",
    )

    candidate_accuracy = accuracy_score(
        y_true,
        candidate_pred,
    )

    candidate_macro = f1_score(
        y_true,
        candidate_pred,
        average="macro",
    )

    candidate_weighted = f1_score(
        y_true,
        candidate_pred,
        average="weighted",
    )

    print(
        f"\nWORD_BIGRAMS"
    )

    print(
        f"Accuracy:     {baseline_accuracy:.4f}"
    )

    print(
        f"Macro F1:     {baseline_macro:.4f}"
    )

    print(
        f"Weighted F1:  {baseline_weighted:.4f}"
    )

    print(
        f"\nWORD_PLUS_CHAR"
    )

    print(
        f"Accuracy:     {candidate_accuracy:.4f}"
    )

    print(
        f"Macro F1:     {candidate_macro:.4f}"
    )

    print(
        f"Weighted F1:  {candidate_weighted:.4f}"
    )

    print(
        f"\nMacro F1 change: "
        f"{candidate_macro - baseline_macro:+.4f}"
    )

    show_confusion_matrix(
        y_true,
        baseline_pred,
        labels,
        "WORD_BIGRAMS CONFUSION MATRIX",
    )

    show_confusion_matrix(
        y_true,
        candidate_pred,
        labels,
        "WORD_PLUS_CHAR CONFUSION MATRIX",
    )

    show_confusion_pairs(
        y_true,
        baseline_pred,
        labels,
        "WORD_BIGRAMS MOST COMMON CONFUSION PAIRS",
    )

    show_confusion_pairs(
        y_true,
        candidate_pred,
        labels,
        "WORD_PLUS_CHAR MOST COMMON CONFUSION PAIRS",
    )

    compare_confusion_changes(
        y_true,
        baseline_pred,
        candidate_pred,
        labels,
    )

    print_section(
        "WORD_BIGRAMS CLASSIFICATION REPORT"
    )

    print(
        classification_report(
            y_true,
            baseline_pred,
            labels=labels,
            zero_division=0,
        )
    )

    print_section(
        "WORD_PLUS_CHAR CLASSIFICATION REPORT"
    )

    print(
        classification_report(
            y_true,
            candidate_pred,
            labels=labels,
            zero_division=0,
        )
    )

    print_section(
        "INSPECTION COMPLETE"
    )

    print(
        "\nNo production dataset was modified."
    )

    print(
        "No production training code was modified."
    )


if __name__ == "__main__":
    main()