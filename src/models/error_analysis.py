import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/validation.csv"
OUTPUT_PATH = "data/processed/validation_errors.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

X_train = train_df["text"].fillna("")
y_train = train_df["intent"]

X_val = val_df["text"].fillna("")
y_val = val_df["intent"]


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)


# --------------------------------------------------
# Train model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(X_val_vec)

accuracy = accuracy_score(
    y_val,
    predictions
)

print("=" * 80)
print("VALIDATION ERROR ANALYSIS")
print("=" * 80)

print(f"Validation accuracy: {accuracy:.4f}")


# --------------------------------------------------
# Build error dataframe
# --------------------------------------------------

error_df = pd.DataFrame({
    "text": X_val,
    "actual": y_val,
    "predicted": predictions
})

error_df = error_df[
    error_df["actual"] != error_df["predicted"]
].copy()


# --------------------------------------------------
# Print errors
# --------------------------------------------------

print("\n" + "=" * 80)
print(f"WRONG PREDICTIONS: {len(error_df)}")
print("=" * 80)

for _, row in error_df.iterrows():

    print("\nTEXT:")
    print(row["text"])

    print(f"ACTUAL:    {row['actual']}")
    print(f"PREDICTED: {row['predicted']}")

    print("-" * 80)


# --------------------------------------------------
# Error pairs
# --------------------------------------------------

print("\n" + "=" * 80)
print("ERROR PAIRS")
print("=" * 80)

error_pairs = (
    error_df
    .groupby(["actual", "predicted"])
    .size()
    .sort_values(ascending=False)
)

print(error_pairs)


# --------------------------------------------------
# Save errors
# --------------------------------------------------

error_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)