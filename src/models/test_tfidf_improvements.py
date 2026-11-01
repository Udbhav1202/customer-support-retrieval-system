from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "data/processed/merged_training_dataset_cleaned.csv"

TEXT_COL = "text"
TARGET_COL = "intent"

RANDOM_STATE = 42
N_SPLITS = 5


# Same rare classes excluded in our previous experiments
RARE_CLASSES = {
    "FRAUD_SECURITY",
    "MOBILE_PHONE_SERVICE",
    "REFUND",
    "SIM_CARD",
}


# Our tested 6-class taxonomy
MERGE_MAP = {
    "CUSTOMER_SUPPORT_ACCESS": "CUSTOMER_SUPPORT",
    "CUSTOMER_SUPPORT_SERVICE_FEEDBACK": "CUSTOMER_SUPPORT",
    "SLOW_INTERNET": "NETWORK_CONNECTIVITY",
    "INTERNET_OUTAGE": "NETWORK_CONNECTIVITY",
}


# ============================================================
# EXPERIMENTS
# ============================================================

EXPERIMENTS = {

    # Current baseline
    "BASELINE": {
        "ngram_range": (1, 1),
        "sublinear_tf": False,
        "min_df": 1,
        "max_df": 1.0,
        "class_weight": "balanced",
    },

    # Add word bigrams
    "WORD_BIGRAMS": {
        "ngram_range": (1, 2),
        "sublinear_tf": False,
        "min_df": 1,
        "max_df": 1.0,
        "class_weight": "balanced",
    },

    # Bigrams + sublinear TF
    "SUBLINEAR_BIGRAMS": {
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 1,
        "max_df": 1.0,
        "class_weight": "balanced",
    },

    # Remove extremely rare terms
    "BIGRAMS_MIN_DF_2": {
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 2,
        "max_df": 1.0,
        "class_weight": "balanced",
    },

    # Same as above but without class balancing
    "BIGRAMS_MIN_DF_2_UNBALANCED": {
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 2,
        "max_df": 1.0,
        "class_weight": None,
    },
}


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    path = Path(CSV_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{path.resolve()}\n"
            f"\nCheck CSV_PATH at the top of the script."
        )

    df = pd.read_csv(path)

    required_columns = {TEXT_COL, TARGET_COL}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"\nMissing columns: {missing}"
            f"\nAvailable columns: {list(df.columns)}"
        )

    df = df.dropna(
        subset=[TEXT_COL, TARGET_COL]
    ).copy()

    # Remove rare classes exactly as previous experiments
    df = df[
        ~df[TARGET_COL].isin(RARE_CLASSES)
    ].copy()

    # Apply 6-class taxonomy
    df[TARGET_COL] = df[TARGET_COL].replace(
        MERGE_MAP
    )

    df[TEXT_COL] = df[TEXT_COL].astype(str)

    return df


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(config):

    vectorizer = TfidfVectorizer(
        ngram_range=config["ngram_range"],
        sublinear_tf=config["sublinear_tf"],
        min_df=config["min_df"],
        max_df=config["max_df"],
        lowercase=True,
    )

    classifier = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight=config["class_weight"],
    )

    model = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])

    return model


# ============================================================
# EVALUATE ONE EXPERIMENT
# ============================================================

def evaluate_experiment(df, name, config):

    X = df[TEXT_COL]
    y = df[TARGET_COL]

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = build_model(config)

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict",
        n_jobs=-1,
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
    )

    return (
        accuracy,
        macro_f1,
        weighted_f1,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TF-IDF IMPROVEMENT SIDE EXPERIMENT")
    print("=" * 80)

    df = load_dataset()

    print(
        f"\nEvaluation dataset: {df.shape}"
    )

    print("\n6-class distribution:")

    print(
        df[TARGET_COL].value_counts()
    )

    print("\n" + "=" * 80)
    print("RUNNING 5-FOLD CV")
    print("=" * 80)

    results = []

    for name, config in EXPERIMENTS.items():

        print(
            f"\nRunning: {name}"
        )

        print(
            f"  ngram_range : "
            f"{config['ngram_range']}"
        )

        print(
            f"  sublinear_tf: "
            f"{config['sublinear_tf']}"
        )

        print(
            f"  min_df      : "
            f"{config['min_df']}"
        )

        print(
            f"  max_df      : "
            f"{config['max_df']}"
        )

        print(
            f"  class_weight: "
            f"{config['class_weight']}"
        )

        accuracy, macro_f1, weighted_f1 = (
            evaluate_experiment(
                df,
                name,
                config,
            )
        )

        print(
            f"  Accuracy    : "
            f"{accuracy:.4f}"
        )

        print(
            f"  Macro F1    : "
            f"{macro_f1:.4f}"
        )

        print(
            f"  Weighted F1 : "
            f"{weighted_f1:.4f}"
        )

        results.append({
            "Experiment": name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
        })

    # ========================================================
    # COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    baseline = results_df.iloc[0]

    results_df["Accuracy Change"] = (
        results_df["Accuracy"]
        - baseline["Accuracy"]
    )

    results_df["Macro F1 Change"] = (
        results_df["Macro F1"]
        - baseline["Macro F1"]
    )

    results_df["Weighted F1 Change"] = (
        results_df["Weighted F1"]
        - baseline["Weighted F1"]
    )

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 80)
    print("IMPORTANT")
    print("=" * 80)

    print(
        "This is a SIDE EXPERIMENT."
    )

    print(
        "No production/training code was modified."
    )

    print(
        "Compare Macro F1 first."
    )

    print(
        "Then inspect the confusion matrix of promising configurations."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()