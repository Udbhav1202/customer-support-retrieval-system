
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


# ================================================================
# CONFIGURATION
# ================================================================

CSV_PATH = "data/processed/merged_training_dataset_cleaned.csv"

RANDOM_STATE = 42
N_SPLITS = 5

TEXT_COL = "text"
INTENT_COL = "intent"

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


# ================================================================
# TAXONOMY
# ================================================================

def apply_6class_taxonomy(intent):

    if intent in NETWORK_CLASSES:
        return "NETWORK_CONNECTIVITY"

    if intent in SUPPORT_CLASSES:
        return "CUSTOMER_SUPPORT"

    return intent


# ================================================================
# MODEL
# ================================================================

def build_model():

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


# ================================================================
# DATA PREPARATION
# ================================================================

def load_dataset():

    df = pd.read_csv(CSV_PATH)

    print(f"Original dataset: {df.shape}")

    # Remove pseudo-labeled rows if present.
    if "is_pseudo_labeled" in df.columns:

        before = len(df)

        df = df[
            df["is_pseudo_labeled"] != True
        ].copy()

        print(
            f"Removed pseudo-labeled rows: "
            f"{before - len(df)}"
        )

    # Remove rare classes exactly as previous experiments.
    before = len(df)

    df = df[
        ~df[INTENT_COL].isin(RARE_CLASSES)
    ].copy()

    print(
        f"Removed rare-class rows: "
        f"{before - len(df)}"
    )

    # Create 6-class taxonomy.
    df["six_class_intent"] = (
        df[INTENT_COL]
        .apply(apply_6class_taxonomy)
    )

    # Clean text.
    df[TEXT_COL] = (
        df[TEXT_COL]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Useful text statistics.
    df["word_count"] = (
        df[TEXT_COL]
        .str.split()
        .str.len()
    )

    return df


# ================================================================
# DATASET VARIANTS
# ================================================================

def create_variants(df):

    variants = {}

    # ------------------------------------------------------------
    # A. CURRENT DATASET
    # ------------------------------------------------------------

    variants["CURRENT"] = df.copy()

    # ------------------------------------------------------------
    # B. REMOVE CLEARLY VERY SHORT / LOW INFORMATION TEXT
    #
    # We use <= 2 words rather than <= 5.
    #
    # This avoids deleting useful short examples such as:
    # "My internet is not working."
    # ------------------------------------------------------------

    short_removed = df[
        df["word_count"] > 2
    ].copy()

    variants["SHORT_REMOVED"] = short_removed

    # ------------------------------------------------------------
    # C. REMOVE SHORT TEXT + SUSPICIOUS OTHER
    #
    # For OTHER we remove only examples that are both:
    #
    #   - OTHER
    #   - <= 6 words
    #
    # This is deliberately conservative.
    # ------------------------------------------------------------

    short_plus_other = df[
        ~(
            (df["word_count"] <= 2)
        )
    ].copy()

    suspicious_other_mask = (
        (short_plus_other["six_class_intent"] == "OTHER")
        & (short_plus_other["word_count"] <= 6)
    )

    short_plus_other = short_plus_other[
        ~suspicious_other_mask
    ].copy()

    variants["SHORT_PLUS_OTHER"] = short_plus_other

    return variants


# ================================================================
# 5-FOLD CROSS VALIDATION
# ================================================================

def evaluate_dataset(df):

    X = df[TEXT_COL]
    y = df["six_class_intent"]

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_accuracy = []
    fold_macro_f1 = []
    fold_weighted_f1 = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = build_model()

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

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

        fold_accuracy.append(accuracy)
        fold_macro_f1.append(macro_f1)
        fold_weighted_f1.append(weighted_f1)

        print(
            f"  Fold {fold}: "
            f"Accuracy={accuracy:.4f} | "
            f"Macro F1={macro_f1:.4f} | "
            f"Weighted F1={weighted_f1:.4f}"
        )

    return {
        "accuracy_mean": sum(fold_accuracy) / N_SPLITS,
        "accuracy_std": pd.Series(fold_accuracy).std(),

        "macro_f1_mean": sum(fold_macro_f1) / N_SPLITS,
        "macro_f1_std": pd.Series(fold_macro_f1).std(),

        "weighted_f1_mean": sum(fold_weighted_f1) / N_SPLITS,
        "weighted_f1_std": pd.Series(fold_weighted_f1).std(),
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 80)
    print("CLEAN DATA SIDE EXPERIMENT")
    print("=" * 80)

    df = load_dataset()

    print("\n6-class distribution:")
    print(
        df["six_class_intent"]
        .value_counts()
    )

    variants = create_variants(df)

    results = []

    print("\n")
    print("=" * 80)
    print("RUNNING 5-FOLD CROSS VALIDATION")
    print("=" * 80)

    for name, variant_df in variants.items():

        print("\n")
        print("-" * 80)

        print(
            f"EXPERIMENT: {name}"
        )

        print("-" * 80)

        print(
            f"Dataset size: "
            f"{len(variant_df)}"
        )

        print("\nClass distribution:")

        print(
            variant_df[
                "six_class_intent"
            ].value_counts()
        )

        metrics = evaluate_dataset(
            variant_df
        )

        results.append({
            "Experiment": name,
            "Dataset Size": len(variant_df),

            "Accuracy": metrics[
                "accuracy_mean"
            ],

            "Accuracy Std": metrics[
                "accuracy_std"
            ],

            "Macro F1": metrics[
                "macro_f1_mean"
            ],

            "Macro F1 Std": metrics[
                "macro_f1_std"
            ],

            "Weighted F1": metrics[
                "weighted_f1_mean"
            ],

            "Weighted F1 Std": metrics[
                "weighted_f1_std"
            ],
        })

    # ============================================================
    # FINAL COMPARISON
    # ============================================================

    results_df = pd.DataFrame(
        results
    )

    baseline_macro_f1 = (
        results_df.loc[
            results_df["Experiment"] == "CURRENT",
            "Macro F1"
        ].iloc[0]
    )

    baseline_accuracy = (
        results_df.loc[
            results_df["Experiment"] == "CURRENT",
            "Accuracy"
        ].iloc[0]
    )

    baseline_weighted_f1 = (
        results_df.loc[
            results_df["Experiment"] == "CURRENT",
            "Weighted F1"
        ].iloc[0]
    )

    results_df[
        "Macro F1 Change"
    ] = (
        results_df["Macro F1"]
        - baseline_macro_f1
    )

    results_df[
        "Accuracy Change"
    ] = (
        results_df["Accuracy"]
        - baseline_accuracy
    )

    results_df[
        "Weighted F1 Change"
    ] = (
        results_df["Weighted F1"]
        - baseline_weighted_f1
    )

    print("\n")
    print("=" * 80)
    print("FINAL CLEAN DATA COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ============================================================
    # INTERPRETATION
    # ============================================================

    print("\n")
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print(
        "\nCURRENT is our baseline."
    )

    print(
        "\nCompare Macro F1 first."
    )

    print(
        "Positive Macro F1 Change = improvement."
    )

    print(
        "Negative Macro F1 Change = degradation."
    )

    print(
        "\nAlso check the standard deviation."
    )

    print(
        "A large increase in mean F1 with a large "
        "increase in fold-to-fold variance needs caution."
    )

    print(
        "\nNo production dataset was modified."
    )

    print(
        "No production training code was modified."
    )

    print(
        "This experiment only compares alternative "
        "training datasets."
    )

    print("\n")
    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
