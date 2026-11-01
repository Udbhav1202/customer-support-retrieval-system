import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


CSV_PATH = "data/processed/merged_training_dataset_cleaned.csv"
TEXT_COL = "text"


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


def prepare_dataset():
    df = pd.read_csv(CSV_PATH)

    print(f"Original dataset: {df.shape}")

    if "is_pseudo_labeled" in df.columns:
        df = df[df["is_pseudo_labeled"] != True].copy()

    df = df[~df["intent"].isin(RARE_CLASSES)].copy()

    df["six_class_intent"] = df["intent"].apply(
        apply_6class_taxonomy
    )

    return df


def remove_ambiguous_other(df):
    """
    Remove ONLY clearly ambiguous OTHER examples.

    These are examples that look like support requests but
    are currently labeled OTHER.
    """

    ambiguous_texts = {
        "@VerizonSupport need help with my fios",
        "@VerizonSupport Need help with an order",
        "@115990 been on the phone all day with Verizon.. Incompetent.. Need help",
        "@VerizonSupport Is anyone else having trouble with verizon",
    }

    mask = (
        (df["six_class_intent"] == "OTHER")
        & (df[TEXT_COL].isin(ambiguous_texts))
    )

    removed = df[mask].copy()
    cleaned = df[~mask].copy()

    return cleaned, removed


def run_cv(df, experiment_name):
    print("\n" + "-" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("-" * 80)

    print(f"Dataset size: {len(df)}")

    print("\nClass distribution:")
    print(df["six_class_intent"].value_counts())

    X = df[TEXT_COL].fillna("")
    y = df["six_class_intent"]

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    accuracies = []
    macro_f1s = []
    weighted_f1s = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )

        model.fit(X_train_vec, y_train)

        predictions = model.predict(X_test_vec)

        accuracy = accuracy_score(
            y_test,
            predictions,
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

        accuracies.append(accuracy)
        macro_f1s.append(macro_f1)
        weighted_f1s.append(weighted_f1)

        print(
            f"  Fold {fold}: "
            f"Accuracy={accuracy:.4f} | "
            f"Macro F1={macro_f1:.4f} | "
            f"Weighted F1={weighted_f1:.4f}"
        )

    return {
        "Experiment": experiment_name,
        "Dataset Size": len(df),
        "Accuracy": np.mean(accuracies),
        "Accuracy Std": np.std(accuracies),
        "Macro F1": np.mean(macro_f1s),
        "Macro F1 Std": np.std(macro_f1s),
        "Weighted F1": np.mean(weighted_f1s),
        "Weighted F1 Std": np.std(weighted_f1s),
    }


def main():

    print("=" * 80)
    print("TARGETED OTHER CLEANUP SIDE EXPERIMENT")
    print("=" * 80)

    df = prepare_dataset()

    print("\n6-class distribution:")
    print(df["six_class_intent"].value_counts())

    # ---------------------------------------------------------------
    # CURRENT BASELINE
    # ---------------------------------------------------------------

    baseline_result = run_cv(
        df,
        "CURRENT",
    )

    # ---------------------------------------------------------------
    # TARGETED OTHER CLEANUP
    # ---------------------------------------------------------------

    cleaned_df, removed_df = remove_ambiguous_other(df)

    print("\n" + "=" * 80)
    print("REMOVED AMBIGUOUS OTHER EXAMPLES")
    print("=" * 80)

    print(f"\nRemoved: {len(removed_df)}")

    for _, row in removed_df.iterrows():
        print("\n" + "-" * 80)
        print(f"Conversation ID: {row['conversation_id']}")
        print(f"Original intent: {row['intent']}")
        print(f"Text: {row[TEXT_COL]}")

    cleaned_result = run_cv(
        cleaned_df,
        "TARGETED_OTHER_REMOVED",
    )

    # ---------------------------------------------------------------
    # COMPARISON
    # ---------------------------------------------------------------

    results = pd.DataFrame(
        [
            baseline_result,
            cleaned_result,
        ]
    )

    baseline_macro = baseline_result["Macro F1"]

    results["Macro F1 Change"] = (
        results["Macro F1"] - baseline_macro
    )

    results["Accuracy Change"] = (
        results["Accuracy"]
        - baseline_result["Accuracy"]
    )

    results["Weighted F1 Change"] = (
        results["Weighted F1"]
        - baseline_result["Weighted F1"]
    )

    print("\n" + "=" * 80)
    print("FINAL OTHER CLEANUP COMPARISON")
    print("=" * 80)

    print(
        results[
            [
                "Experiment",
                "Dataset Size",
                "Accuracy",
                "Accuracy Std",
                "Macro F1",
                "Macro F1 Std",
                "Weighted F1",
                "Macro F1 Change",
                "Accuracy Change",
                "Weighted F1 Change",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print(
        """
CURRENT is the baseline.

Compare Macro F1 first.

Positive Macro F1 Change:
    Potential improvement.

Negative Macro F1 Change:
    Potential degradation.

This is a SIDE EXPERIMENT.

No production dataset was modified.
No production training code was modified.
"""
    )


if __name__ == "__main__":
    main()