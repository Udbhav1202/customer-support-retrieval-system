import html
import re

import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


DATA_PATH = "data/processed/merged_training_dataset_cleaned.csv"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3


# ============================================================
# TEXT CLEANING
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


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("EXPERIMENT 10: FINAL CLASSIFIER ROBUSTNESS")
print("=" * 80)

print(f"\nOriginal dataset: {df.shape}")


# ============================================================
# REMOVE PSEUDO-LABELED ROWS
# ============================================================

if "is_pseudo_labeled" in df.columns:
    df = df[df["is_pseudo_labeled"] != True].copy()


# ============================================================
# 6-CLASS TAXONOMY
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
# PREPARE DATA
# ============================================================

X_text = df["text"].fillna("").apply(clean_text)
y = df["six_class_intent"]


print(f"Evaluation dataset: {df.shape}")

print("\n6-class distribution:")
print(y.value_counts())


# ============================================================
# FIXED FEATURE REPRESENTATION
# Current best from Experiments 8 and 9:
#
# Word TF-IDF:
#   (1,2), min_df=2, sublinear_tf=True
#
# Character TF-IDF:
#   (2,4), min_df=1, sublinear_tf=True
# ============================================================

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    min_df=1,
    sublinear_tf=True,
)

X_word = word_vectorizer.fit_transform(X_text)
X_char = char_vectorizer.fit_transform(X_text)

X = hstack([
    X_word,
    X_char
]).tocsr()


print(f"\nWord feature dimensions: {X_word.shape[1]}")
print(f"Char feature dimensions: {X_char.shape[1]}")
print(f"Combined feature dimensions: {X.shape[1]}")


# ============================================================
# REPEATED STRATIFIED CROSS-VALIDATION
# 5 folds × 3 repeats = 15 evaluation folds
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
)


# ============================================================
# CLASSIFIERS
# ============================================================

classifiers = {
    "LINEAR_SVM": LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "LOGISTIC_REGRESSION": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),
}


results = []


# ============================================================
# RUN ROBUSTNESS EXPERIMENT
# ============================================================

for classifier_name, classifier in classifiers.items():

    print("\n" + "=" * 80)
    print(classifier_name)
    print("=" * 80)

    fold_scores = []

    for fold_number, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print(
            f"Fold {fold_number}/{N_SPLITS * N_REPEATS}..."
        )

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        classifier.fit(
            X_train,
            y_train,
        )

        predictions = classifier.predict(
            X_test
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
        )

        fold_scores.append(macro_f1)

    fold_scores = np.array(fold_scores)

    results.append({
        "classifier": classifier_name,
        "mean_macro_f1": fold_scores.mean(),
        "std_macro_f1": fold_scores.std(),
        "min_macro_f1": fold_scores.min(),
        "max_macro_f1": fold_scores.max(),
    })

    print(
        f"\nMean Macro F1: {fold_scores.mean():.4f}"
    )
    print(
        f"Std Macro F1:  {fold_scores.std():.4f}"
    )
    print(
        f"Min Macro F1:  {fold_scores.min():.4f}"
    )
    print(
        f"Max Macro F1:  {fold_scores.max():.4f}"
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "mean_macro_f1",
    ascending=False,
)


print("\n" + "=" * 80)
print("FINAL ROBUSTNESS RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        formatters={
            "mean_macro_f1": "{:.4f}".format,
            "std_macro_f1": "{:.4f}".format,
            "min_macro_f1": "{:.4f}".format,
            "max_macro_f1": "{:.4f}".format,
        },
    )
)


# ============================================================
# HEAD-TO-HEAD
# ============================================================

svm_score = results_df.loc[
    results_df["classifier"] == "LINEAR_SVM",
    "mean_macro_f1"
].iloc[0]

lr_score = results_df.loc[
    results_df["classifier"] == "LOGISTIC_REGRESSION",
    "mean_macro_f1"
].iloc[0]

print("\n" + "=" * 80)
print("SVM vs LOGISTIC REGRESSION")
print("=" * 80)

print(f"Linear SVM:          {svm_score:.4f}")
print(f"Logistic Regression: {lr_score:.4f}")
print(f"Difference:          {svm_score - lr_score:+.4f}")


# ============================================================
# SAVE
# ============================================================

output_path = (
    "data/analysis/final_classifier_robustness_results.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

print(
    f"\nResults saved to: {output_path}"
)

print("\nNo production dataset was modified.")
print("No production training code was modified.")