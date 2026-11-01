import pandas as pd
import numpy as np
import html
import re
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


DATA_PATH = "data/processed/merged_training_dataset_cleaned.csv"

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("EXPERIMENT 8: CHARACTER N-GRAM TUNING")
print("=" * 80)

print(f"\nOriginal dataset: {df.shape}")


# ============================================================
# REMOVE PSEUDO-LABELED ROWS
# ============================================================

if "is_pseudo_labeled" in df.columns:
    df = df[df["is_pseudo_labeled"] != True].copy()


# ============================================================
# MAP TO 6-CLASS TAXONOMY
# ============================================================

def map_to_six_class(intent):
    mapping = {
        "CUSTOMER_SUPPORT_ACCESS": "CUSTOMER_SUPPORT",
        "CUSTOMER_SUPPORT_SERVICE_FEEDBACK": "CUSTOMER_SUPPORT",

        "SLOW_INTERNET": "NETWORK_CONNECTIVITY",
        "INTERNET_OUTAGE": "NETWORK_CONNECTIVITY",

        "SWITCHING_RETENTION": "SWITCHING_RETENTION",
        "BILLING_PAYMENT": "BILLING_PAYMENT",
        "WIFI_ROUTER": "WIFI_ROUTER",
        "OTHER": "OTHER",
    }

    return mapping.get(intent)


df["six_class_intent"] = df["intent"].map(map_to_six_class)

df = df[df["six_class_intent"].notna()].copy()


# ============================================================
# REMOVE RARE CLASSES
# Same filtering used in previous experiments
# ============================================================

rare_classes = [
    "FRAUD_SECURITY",
    "MOBILE_PHONE_SERVICE",
    "REFUND",
    "SIM_CARD",
]

if "intent" in df.columns:
    df = df[~df["intent"].isin(rare_classes)].copy()


# ============================================================
# PREPARE DATA
# ============================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


X_text = df["text"].fillna("").apply(clean_text)
y = df["six_class_intent"]


print(f"Evaluation dataset: {df.shape}")

print("\n6-class distribution:")
print(y.value_counts())


# ============================================================
# FIXED WORD FEATURES
# Same configuration as Experiment 7
# ============================================================

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

X_word = word_vectorizer.fit_transform(X_text)


# ============================================================
# CHARACTER CONFIGURATIONS
# ============================================================

char_configs = [
    ("CHAR_2_4", (2, 4)),
    ("CHAR_2_5", (2, 5)),
    ("CHAR_3_5", (3, 5)),
    ("CHAR_3_6", (3, 6)),
    ("CHAR_2_6", (2, 6)),
]


# ============================================================
# STRATIFIED CV
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


results = []


# ============================================================
# RUN EXPERIMENTS
# ============================================================

for config_name, char_range in char_configs:

    print("\n" + "=" * 80)
    print(config_name)
    print(f"Character n-gram range: {char_range}")
    print("=" * 80)

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=char_range,
        min_df=1,
        sublinear_tf=True,
    )

    X_char = char_vectorizer.fit_transform(X_text)

    X_combined = hstack([
        X_word,
        X_char
    ]).tocsr()

    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X_combined, y),
        start=1
    ):

        print(f"Fold {fold}/{N_SPLITS}...")

        X_train = X_combined[train_idx]
        X_test = X_combined[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
        )

        fold_scores.append(macro_f1)

    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)

    results.append({
        "configuration": config_name,
        "char_ngram_range": str(char_range),
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
    })

    print(f"\nMean Macro F1: {mean_f1:.4f}")
    print(f"Std Macro F1:  {std_f1:.4f}")


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "mean_macro_f1",
    ascending=False,
)


print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        formatters={
            "mean_macro_f1": "{:.4f}".format,
            "std_macro_f1": "{:.4f}".format,
        },
    )
)


# ============================================================
# BASELINE COMPARISON
# Experiment 7 Word + Char (2,5)
# ============================================================

baseline = 0.6901

best_score = results_df.iloc[0]["mean_macro_f1"]
best_config = results_df.iloc[0]["configuration"]

print("\n" + "=" * 80)
print("COMPARISON WITH EXPERIMENT 7")
print("=" * 80)

print(f"Experiment 7 CHAR_2_5: {baseline:.4f}")
print(f"Best configuration:    {best_config}")
print(f"Best Macro F1:         {best_score:.4f}")
print(f"Change:                {best_score - baseline:+.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = "data/analysis/char_ngram_tuning_results.csv"

results_df.to_csv(
    output_path,
    index=False,
)

print(f"\nResults saved to: {output_path}")

print("\nNo production dataset was modified.")
print("No production training code was modified.")