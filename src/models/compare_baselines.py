import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/validation.csv"
TEST_PATH = "data/processed/test.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["text"].fillna("")
y_train = train_df["intent"]

X_val = val_df["text"].fillna("")
y_val = val_df["intent"]

X_test = test_df["text"].fillna("")
y_test = test_df["intent"]


# --------------------------------------------------
# Different TF-IDF configurations
# --------------------------------------------------

experiments = [

    {
        "name": "Word TF-IDF (1,2)",
        "vectorizer": TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )
    },

    {
        "name": "Word TF-IDF (1,3)",
        "vectorizer": TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 3),
            min_df=1
        )
    },

    {
        "name": "Word TF-IDF no stopwords",
        "vectorizer": TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1
        )
    },

    {
        "name": "Character TF-IDF",
        "vectorizer": TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1
        )
    }
]


# --------------------------------------------------
# Run experiments
# --------------------------------------------------

results = []


for experiment in experiments:

    print("\n" + "=" * 80)
    print(experiment["name"])
    print("=" * 80)

    vectorizer = experiment["vectorizer"]

    X_train_vec = vectorizer.fit_transform(X_train)

    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    print("Features:", X_train_vec.shape[1])

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    model.fit(
        X_train_vec,
        y_train
    )

    val_pred = model.predict(X_val_vec)
    test_pred = model.predict(X_test_vec)

    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    val_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )

    print("Validation accuracy:", round(val_accuracy, 4))
    print("Validation macro F1:", round(val_f1, 4))

    print("Test accuracy:", round(test_accuracy, 4))
    print("Test macro F1:", round(test_f1, 4))

    results.append({
        "model": experiment["name"],
        "features": X_train_vec.shape[1],
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_f1,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_f1
    })


# --------------------------------------------------
# Summary
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("BASELINE COMPARISON")
print("=" * 80)

print(
    results_df.to_string(index=False)
)