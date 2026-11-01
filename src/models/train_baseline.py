import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/validation.csv"
TEST_PATH = "data/processed/test.csv"


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


# --------------------------------------------------
# Separate text and labels
# --------------------------------------------------

X_train = train_df["text"].fillna("")
y_train = train_df["intent"]

X_val = val_df["text"].fillna("")
y_val = val_df["intent"]

X_test = test_df["text"].fillna("")
y_test = test_df["intent"]


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF:")
print("Training shape:", X_train_tfidf.shape)
print("Validation shape:", X_val_tfidf.shape)
print("Test shape:", X_test_tfidf.shape)


# --------------------------------------------------
# Train Logistic Regression
# --------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(
    X_train_tfidf,
    y_train
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

val_predictions = model.predict(X_val_tfidf)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

print("Accuracy:", round(val_accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        val_predictions,
        zero_division=0
    )
)


# --------------------------------------------------
# Test
# --------------------------------------------------

test_predictions = model.predict(X_test_tfidf)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("\n" + "=" * 80)
print("TEST RESULTS")
print("=" * 80)

print("Accuracy:", round(test_accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)