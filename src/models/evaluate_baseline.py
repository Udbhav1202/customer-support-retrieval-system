import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


X_train = train_df["text"].fillna("")
y_train = train_df["intent"]

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
X_test_tfidf = vectorizer.transform(X_test)


# --------------------------------------------------
# Train model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(X_test_tfidf)


# --------------------------------------------------
# Show individual predictions
# --------------------------------------------------

print("\n" + "=" * 80)
print("TEST PREDICTIONS")
print("=" * 80)

for i in range(len(test_df)):

    text = test_df.iloc[i]["text"]

    actual = y_test.iloc[i]
    predicted = predictions[i]

    print("\nTEXT:")
    print(text)

    print("ACTUAL:   ", actual)
    print("PREDICTED:", predicted)

    if actual == predicted:
        print("RESULT: CORRECT")
    else:
        print("RESULT: WRONG")


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print("\nLabels:")
print(labels)

print("\nMatrix:")
print(cm)


# --------------------------------------------------
# Plot
# --------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(12, 10))

disp.plot(
    ax=ax,
    xticks_rotation=90
)

plt.tight_layout()

plt.savefig(
    "data/processed/confusion_matrix.png"
)

plt.show()

print("\nSaved confusion matrix:")
print("data/processed/confusion_matrix.png")