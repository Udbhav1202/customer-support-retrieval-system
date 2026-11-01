import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.sparse import hstack


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

    df = df.dropna(subset=[TEXT_COL, "six_class_intent"]).copy()

    return df


def build_vectorizers():
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

    return word_vectorizer, char_vectorizer


def run_experiment(df, experiment_name, splits):
    texts = df[TEXT_COL].astype(str).values
    labels = df["six_class_intent"].values

    accuracy_scores = []
    macro_f1_scores = []
    weighted_f1_scores = []

    print("\n" + "-" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("-" * 80)

    for fold, (train_idx, test_idx) in enumerate(
        splits.split(texts, labels),
        start=1,
    ):
        X_train_text = texts[train_idx]
        X_test_text = texts[test_idx]

        y_train = labels[train_idx]
        y_test = labels[test_idx]

        word_vectorizer, char_vectorizer = build_vectorizers()

        if experiment_name == "WORD_BIGRAMS":
            X_train = word_vectorizer.fit_transform(X_train_text)
            X_test = word_vectorizer.transform(X_test_text)

        elif experiment_name == "CHAR_2_5":
            X_train = char_vectorizer.fit_transform(X_train_text)
            X_test = char_vectorizer.transform(X_test_text)

        elif experiment_name == "WORD_PLUS_CHAR":
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
                [X_train_word, X_train_char]
            )

            X_test = hstack(
                [X_test_word, X_test_char]
            )

        else:
            raise ValueError(
                f"Unknown experiment: {experiment_name}"
            )

        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
        )

        weighted_f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
        )

        accuracy_scores.append(accuracy)
        macro_f1_scores.append(macro_f1)
        weighted_f1_scores.append(weighted_f1)

        print(
            f"  Fold {fold}: "
            f"Accuracy={accuracy:.4f} | "
            f"Macro F1={macro_f1:.4f} | "
            f"Weighted F1={weighted_f1:.4f}"
        )

    return {
        "Experiment": experiment_name,
        "Accuracy Mean": np.mean(accuracy_scores),
        "Accuracy Std": np.std(accuracy_scores),
        "Macro F1 Mean": np.mean(macro_f1_scores),
        "Macro F1 Std": np.std(macro_f1_scores),
        "Weighted F1 Mean": np.mean(weighted_f1_scores),
        "Weighted F1 Std": np.std(weighted_f1_scores),
    }


def main():

    print("=" * 80)
    print("ROBUST WORD vs CHARACTER TF-IDF EXPERIMENT")
    print("=" * 80)

    df = load_dataset()

    print(f"\nEvaluation dataset: {df.shape}")

    print("\n6-class distribution:")
    print(
        df["six_class_intent"].value_counts()
    )

    print("\n" + "=" * 80)
    print("REPEATED STRATIFIED CROSS VALIDATION")
    print("=" * 80)

    print("\nConfiguration:")
    print("  Splits: 5")
    print("  Repeats: 3")
    print("  Total folds per experiment: 15")

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42,
    )

    experiments = [
        "WORD_BIGRAMS",
        "CHAR_2_5",
        "WORD_PLUS_CHAR",
    ]

    results = []

    for experiment in experiments:
        result = run_experiment(
            df,
            experiment,
            cv,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 80)
    print("FINAL ROBUSTNESS COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    baseline_macro = results_df.loc[
        results_df["Experiment"] == "WORD_BIGRAMS",
        "Macro F1 Mean",
    ].iloc[0]

    results_df["Macro F1 Change"] = (
        results_df["Macro F1 Mean"]
        - baseline_macro
    )

    print("\n")
    print("=" * 80)
    print("MACRO F1 CHANGE VS BASELINE")
    print("=" * 80)

    print(
        results_df[
            [
                "Experiment",
                "Macro F1 Mean",
                "Macro F1 Std",
                "Macro F1 Change",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n")
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print(
        """
This is a SIDE EXPERIMENT.

The current WORD_BIGRAMS configuration is the baseline.

The purpose of this experiment is to determine whether
character TF-IDF and WORD + CHARACTER TF-IDF improvements
remain consistent across multiple train/test splits.

Compare:

1. Mean Macro F1
2. Macro F1 standard deviation
3. Consistency across folds

A configuration should NOT be selected only because it
performed well on one particular split.

No production dataset was modified.
No production training code was modified.
"""
    )

    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()