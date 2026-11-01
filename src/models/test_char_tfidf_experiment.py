import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion


CSV_PATH = "data/processed/merged_training_dataset_cleaned.csv"

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

    df = df[~df["intent"].isin(RARE_CLASSES)].copy()

    df["six_class_intent"] = df["intent"].apply(
        apply_6class_taxonomy
    )

    return df


def evaluate_experiment(name, vectorizer, X, y):
    print("\n" + "-" * 80)
    print(f"EXPERIMENT: {name}")
    print("-" * 80)

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
    )

    pipeline = vectorizer

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    accuracy = results["test_accuracy"].mean()
    macro_f1 = results["test_macro_f1"].mean()
    weighted_f1 = results["test_weighted_f1"].mean()

    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    return {
        "Experiment": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
    }


def main():
    print("=" * 80)
    print("WORD vs CHARACTER TF-IDF SIDE EXPERIMENT")
    print("=" * 80)

    df = load_dataset()

    print(f"\nEvaluation dataset: {df.shape}")

    print("\n6-class distribution:")
    print(df["six_class_intent"].value_counts())

    X = df["text"].fillna("")
    y = df["six_class_intent"]

    experiments = []

    # ------------------------------------------------------------
    # 1. CURRENT BEST WORD TF-IDF
    # ------------------------------------------------------------

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    from sklearn.pipeline import Pipeline

    experiments.append(
        evaluate_experiment(
            "WORD_BIGRAMS_MIN_DF_2",
            Pipeline([
                ("tfidf", word_vectorizer),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                    ),
                ),
            ]),
            X,
            y,
        )
    )

    # ------------------------------------------------------------
    # 2. CHARACTER TF-IDF
    # ------------------------------------------------------------

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
    )

    experiments.append(
        evaluate_experiment(
            "CHAR_2_5",
            Pipeline([
                ("tfidf", char_vectorizer),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                    ),
                ),
            ]),
            X,
            y,
        )
    )

    # ------------------------------------------------------------
    # 3. CHARACTER TF-IDF WITH MIN_DF=2
    # ------------------------------------------------------------

    char_vectorizer_min2 = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        sublinear_tf=True,
    )

    experiments.append(
        evaluate_experiment(
            "CHAR_2_5_MIN_DF_2",
            Pipeline([
                ("tfidf", char_vectorizer_min2),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                    ),
                ),
            ]),
            X,
            y,
        )
    )

    # ------------------------------------------------------------
    # 4. WORD + CHARACTER TF-IDF
    # ------------------------------------------------------------

    combined_vectorizer = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                min_df=1,
                sublinear_tf=True,
            ),
        ),
    ])

    experiments.append(
        evaluate_experiment(
            "WORD_PLUS_CHAR",
            Pipeline([
                ("tfidf", combined_vectorizer),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                    ),
                ),
            ]),
            X,
            y,
        )
    )

    # ------------------------------------------------------------
    # FINAL COMPARISON
    # ------------------------------------------------------------

    results_df = pd.DataFrame(experiments)

    baseline_macro = results_df.loc[
        results_df["Experiment"] == "WORD_BIGRAMS_MIN_DF_2",
        "Macro F1",
    ].iloc[0]

    baseline_accuracy = results_df.loc[
        results_df["Experiment"] == "WORD_BIGRAMS_MIN_DF_2",
        "Accuracy",
    ].iloc[0]

    baseline_weighted = results_df.loc[
        results_df["Experiment"] == "WORD_BIGRAMS_MIN_DF_2",
        "Weighted F1",
    ].iloc[0]

    results_df["Macro F1 Change"] = (
        results_df["Macro F1"] - baseline_macro
    )

    results_df["Accuracy Change"] = (
        results_df["Accuracy"] - baseline_accuracy
    )

    results_df["Weighted F1 Change"] = (
        results_df["Weighted F1"] - baseline_weighted
    )

    print("\n")
    print("=" * 80)
    print("FINAL TF-IDF COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
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
WORD_BIGRAMS_MIN_DF_2 is the baseline.

Compare Macro F1 first.

Positive Macro F1 Change:
    Potential improvement.

Negative Macro F1 Change:
    Potential degradation.

Accuracy and Weighted F1 are secondary metrics.

No production dataset was modified.
No production training code was modified.
This is a side experiment only.
"""
    )


if __name__ == "__main__":
    main()