import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/merged_training_dataset_cleaned.csv"

TEXT_COL = "text"
LABEL_COL = "intent"

RANDOM_STATE = 42
N_SPLITS = 5

TOP_K = 8


# ============================================================
# TAXONOMY
# ============================================================

def merge_labels(label):
    """
    BOTH_MERGES taxonomy:

    SLOW_INTERNET + INTERNET_OUTAGE
        -> NETWORK_CONNECTIVITY

    CUSTOMER_SUPPORT_ACCESS + CUSTOMER_SUPPORT_SERVICE_FEEDBACK
        -> CUSTOMER_SUPPORT
    """

    if label in [
        "SLOW_INTERNET",
        "INTERNET_OUTAGE",
    ]:
        return "NETWORK_CONNECTIVITY"

    if label in [
        "CUSTOMER_SUPPORT_ACCESS",
        "CUSTOMER_SUPPORT_SERVICE_FEEDBACK",
    ]:
        return "CUSTOMER_SUPPORT"

    return label


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    df = pd.read_csv(DATA_PATH)

    print("=" * 80)
    print("6-CLASS CONFUSION INSPECTION")
    print("=" * 80)

    print(f"\nOriginal dataset: {df.shape}")

    # Remove pseudo-labelled data
    if "source" in df.columns:

        pseudo_sources = [
            "tfidf_pseudo_label"
        ]

        before = len(df)

        df = df[
            ~df["source"].isin(pseudo_sources)
        ].copy()

        removed = before - len(df)

        print(
            f"Excluded {removed} rows "
            f"with source in {pseudo_sources}"
        )

    # Keep only classes that are part of the main evaluation
    main_classes = [
        "SWITCHING_RETENTION",
        "BILLING_PAYMENT",
        "CUSTOMER_SUPPORT_SERVICE_FEEDBACK",
        "WIFI_ROUTER",
        "CUSTOMER_SUPPORT_ACCESS",
        "OTHER",
        "INTERNET_OUTAGE",
        "SLOW_INTERNET",
    ]

    rare_classes = sorted(
        set(df[LABEL_COL])
        - set(main_classes)
    )

    if rare_classes:

        print("\nRare classes kept out of evaluation:")

        for cls in rare_classes:

            count = (
                df[LABEL_COL] == cls
            ).sum()

            print(
                f"  {cls}: {count} examples"
            )

        df = df[
            df[LABEL_COL].isin(main_classes)
        ].copy()

    # Apply merged taxonomy
    df["original_intent"] = df[LABEL_COL]

    df[LABEL_COL] = df[
        LABEL_COL
    ].apply(merge_labels)

    print(
        f"\n6-class evaluation dataset: "
        f"{df.shape}"
    )

    print("\nClass distribution:")

    print(
        df[LABEL_COL]
        .value_counts()
    )

    return df


# ============================================================
# OOF PREDICTIONS
# ============================================================

def generate_oof_predictions(df):

    texts = df[TEXT_COL].fillna("").astype(str).values

    y = df[LABEL_COL].values

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predictions = np.empty(
        len(df),
        dtype=object,
    )

    probabilities = np.zeros(
        len(df),
        dtype=float,
    )

    margins = np.zeros(
        len(df),
        dtype=float,
    )

    classes = sorted(
        df[LABEL_COL].unique()
    )

    print("\n")
    print("=" * 80)
    print("GENERATING 5-FOLD OOF PREDICTIONS")
    print("=" * 80)

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(texts, y),
        start=1,
    ):

        print(
            f"\nGenerating fold "
            f"{fold}/{N_SPLITS}..."
        )

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )

        X_train = vectorizer.fit_transform(
            texts[train_idx]
        )

        X_test = vectorizer.transform(
            texts[test_idx]
        )

        # ----------------------------------------------------
        # Logistic Regression
        # ----------------------------------------------------

        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )

        model.fit(
            X_train,
            y[train_idx],
        )

        fold_predictions = model.predict(
            X_test
        )

        fold_probabilities = (
            model.predict_proba(X_test)
        )

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        predictions[test_idx] = (
            fold_predictions
        )

        # Highest probability
        sorted_probs = np.sort(
            fold_probabilities,
            axis=1,
        )

        probabilities[test_idx] = (
            sorted_probs[:, -1]
        )

        # Difference between top two
        margins[test_idx] = (
            sorted_probs[:, -1]
            -
            sorted_probs[:, -2]
        )

    return (
        predictions,
        probabilities,
        margins,
    )


# ============================================================
# CONFUSION PAIRS
# ============================================================

def get_confusion_pairs(
    y_true,
    predictions,
):

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=sorted(
            np.unique(y_true)
        ),
    )

    labels = sorted(
        np.unique(y_true)
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
                        count,
                        actual,
                        predicted,
                    )
                )

    pairs.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return pairs


# ============================================================
# SHOW CONFUSED EXAMPLES
# ============================================================

def show_confused_examples(
    df,
    predictions,
    probabilities,
    margins,
    pairs,
    top_k=TOP_K,
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

    y_true = df[LABEL_COL].values

    for count, actual, predicted in pairs:

        print("\n" + "-" * 80)

        print(
            f"CONFUSION COUNT: {count}"
        )

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

        # Lowest margin first
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

            if "conversation_id" in df.columns:

                print(
                    f"Conversation ID: "
                    f"{df.iloc[index]['conversation_id']}"
                )

            print(
                f"Original intent: "
                f"{df.iloc[index]['original_intent']}"
            )

            print(
                f"Text:\n"
                f"{df.iloc[index][TEXT_COL]}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    (
        predictions,
        probabilities,
        margins,
    ) = generate_oof_predictions(df)

    y_true = df[LABEL_COL].values

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\n")
    print("=" * 80)
    print("6-CLASS OOF PERFORMANCE")
    print("=" * 80)

    print(
        f"\nAccuracy:     {accuracy:.4f}"
    )

    print(
        f"Macro F1:     {macro_f1:.4f}"
    )

    print(
        f"Weighted F1:  {weighted_f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)

    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    labels = sorted(
        np.unique(y_true)
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=labels,
    )

    print("\n")
    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    confusion_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual: {x}"
            for x in labels
        ],
        columns=[
            f"Pred: {x}"
            for x in labels
        ],
    )

    print(
        confusion_df.to_string()
    )

    # --------------------------------------------------------
    # Most common confusion pairs
    # --------------------------------------------------------

    pairs = get_confusion_pairs(
        y_true,
        predictions,
    )

    print("\n")
    print("=" * 80)
    print("MOST COMMON CONFUSION PAIRS")
    print("=" * 80)

    for count, actual, predicted in pairs[:15]:

        print(
            f"{actual:30s}"
            f" -> "
            f"{predicted:30s}"
            f" : {count}"
        )

    # --------------------------------------------------------
    # Confused examples
    # --------------------------------------------------------

    show_confused_examples(
        df,
        predictions,
        probabilities,
        margins,
        pairs[:10],
        top_k=TOP_K,
    )

    print("\n")
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)

    print(
        "\nNo files were modified."
    )


if __name__ == "__main__":
    main()