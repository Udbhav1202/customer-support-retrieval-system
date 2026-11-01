import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack


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


def prepare_dataset():
    df = pd.read_csv(CSV_PATH)

    if "is_pseudo_labeled" in df.columns:
        df = df[df["is_pseudo_labeled"] != True].copy()

    df = df[~df["intent"].isin(RARE_CLASSES)].copy()

    df["six_class_intent"] = df["intent"].apply(
        apply_6class_taxonomy
    )

    return df.reset_index(drop=True)


def build_word_features(train_text, test_text):
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    return X_train, X_test


def build_word_char_features(train_text, test_text):
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

    X_train_word = word_vectorizer.fit_transform(train_text)
    X_test_word = word_vectorizer.transform(test_text)

    X_train_char = char_vectorizer.fit_transform(train_text)
    X_test_char = char_vectorizer.transform(test_text)

    X_train = hstack([
        X_train_word,
        X_train_char,
    ])

    X_test = hstack([
        X_test_word,
        X_test_char,
    ])

    return X_train, X_test


def train_predict(X_train, y_train, X_test):
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model.predict(X_test)


def main():

    print("=" * 80)
    print("NETWORK + OTHER WORD vs WORD+CHAR ERROR ANALYSIS")
    print("=" * 80)

    df = prepare_dataset()

    print(f"\nEvaluation dataset: {df.shape}")

    print("\n6-class distribution:")
    print(df["six_class_intent"].value_counts())

    texts = df["text"].fillna("").astype(str)
    labels = df["six_class_intent"]

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    print("\n" + "=" * 80)
    print("GENERATING OOF PREDICTIONS")
    print("=" * 80)

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(texts, labels),
        start=1,
    ):

        print(f"\nFold {fold}/5...")

        train_text = texts.iloc[train_idx]
        test_text = texts.iloc[test_idx]

        y_train = labels.iloc[train_idx]
        y_test = labels.iloc[test_idx]

        # --------------------------------------------------------------
        # WORD BIGRAMS
        # --------------------------------------------------------------

        X_train_word, X_test_word = build_word_features(
            train_text,
            test_text,
        )

        pred_word = train_predict(
            X_train_word,
            y_train,
            X_test_word,
        )

        # --------------------------------------------------------------
        # WORD + CHARACTER
        # --------------------------------------------------------------

        X_train_wc, X_test_wc = build_word_char_features(
            train_text,
            test_text,
        )

        pred_wc = train_predict(
            X_train_wc,
            y_train,
            X_test_wc,
        )

        for idx, actual, word_pred, wc_pred in zip(
            test_idx,
            y_test,
            pred_word,
            pred_wc,
        ):

            results.append({
                "index": idx,
                "actual": actual,
                "word_pred": word_pred,
                "word_char_pred": wc_pred,
            })

    results_df = pd.DataFrame(results)

    results_df["text"] = results_df["index"].apply(
        lambda i: df.iloc[i]["text"]
    )

    results_df["conversation_id"] = results_df["index"].apply(
        lambda i: (
            df.iloc[i]["conversation_id"]
            if "conversation_id" in df.columns
            else ""
        )
    )

    # ==================================================================
    # NETWORK CONNECTIVITY
    # ==================================================================

    print("\n")
    print("=" * 80)
    print("NETWORK_CONNECTIVITY ANALYSIS")
    print("=" * 80)

    network = results_df[
        results_df["actual"] == "NETWORK_CONNECTIVITY"
    ].copy()

    network_improved = network[
        (network["word_pred"] != network["actual"])
        & (network["word_char_pred"] == network["actual"])
    ]

    network_worsened = network[
        (network["word_pred"] == network["actual"])
        & (network["word_char_pred"] != network["actual"])
    ]

    print(
        f"\nCorrected by WORD+CHAR: "
        f"{len(network_improved)}"
    )

    for _, row in network_improved.iterrows():
        print("\n" + "-" * 80)
        print(f"Conversation ID: {row['conversation_id']}")
        print(f"Actual: {row['actual']}")
        print(f"WORD prediction: {row['word_pred']}")
        print(f"WORD+CHAR prediction: {row['word_char_pred']}")
        print(f"Text: {row['text']}")

    print(
        f"\nBroken by WORD+CHAR: "
        f"{len(network_worsened)}"
    )

    for _, row in network_worsened.iterrows():
        print("\n" + "-" * 80)
        print(f"Conversation ID: {row['conversation_id']}")
        print(f"Actual: {row['actual']}")
        print(f"WORD prediction: {row['word_pred']}")
        print(f"WORD+CHAR prediction: {row['word_char_pred']}")
        print(f"Text: {row['text']}")

    # ==================================================================
    # OTHER
    # ==================================================================

    print("\n")
    print("=" * 80)
    print("OTHER CLASS ANALYSIS")
    print("=" * 80)

    other = results_df[
        results_df["actual"] == "OTHER"
    ].copy()

    other_improved = other[
        (other["word_pred"] != other["actual"])
        & (other["word_char_pred"] == other["actual"])
    ]

    other_worsened = other[
        (other["word_pred"] == other["actual"])
        & (other["word_char_pred"] != other["actual"])
    ]

    print(
        f"\nCorrected by WORD+CHAR: "
        f"{len(other_improved)}"
    )

    for _, row in other_improved.iterrows():
        print("\n" + "-" * 80)
        print(f"Conversation ID: {row['conversation_id']}")
        print(f"Actual: {row['actual']}")
        print(f"WORD prediction: {row['word_pred']}")
        print(f"WORD+CHAR prediction: {row['word_char_pred']}")
        print(f"Text: {row['text']}")

    print(
        f"\nBroken by WORD+CHAR: "
        f"{len(other_worsened)}"
    )

    for _, row in other_worsened.iterrows():
        print("\n" + "-" * 80)
        print(f"Conversation ID: {row['conversation_id']}")
        print(f"Actual: {row['actual']}")
        print(f"WORD prediction: {row['word_pred']}")
        print(f"WORD+CHAR prediction: {row['word_char_pred']}")
        print(f"Text: {row['text']}")

    # ==================================================================
    # SUMMARY
    # ==================================================================

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"\nNETWORK_CONNECTIVITY:"
        f"\n  Corrected: {len(network_improved)}"
        f"\n  Broken:    {len(network_worsened)}"
    )

    print(
        f"\nOTHER:"
        f"\n  Corrected: {len(other_improved)}"
        f"\n  Broken:    {len(other_worsened)}"
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print("\nNo production dataset was modified.")
    print("No production training code was modified.")


if __name__ == "__main__":
    main()