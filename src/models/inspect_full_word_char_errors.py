import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score


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

    df = df.dropna(
        subset=["text", "six_class_intent"]
    ).copy()

    return df.reset_index(drop=True)


def train_word_model(train_text, test_text, y_train):
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model.predict(X_test)


def train_word_char_model(
    train_text,
    test_text,
    y_train,
):
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

    X_train_word = word_vectorizer.fit_transform(
        train_text
    )

    X_test_word = word_vectorizer.transform(
        test_text
    )

    X_train_char = char_vectorizer.fit_transform(
        train_text
    )

    X_test_char = char_vectorizer.transform(
        test_text
    )

    X_train = hstack([
        X_train_word,
        X_train_char,
    ])

    X_test = hstack([
        X_test_word,
        X_test_char,
    ])

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model.predict(X_test)


def main():

    print("=" * 80)
    print("FULL WORD vs WORD+CHAR ERROR ANALYSIS")
    print("=" * 80)

    df = load_dataset()

    print(
        f"\nEvaluation dataset: {df.shape}"
    )

    print("\n6-class distribution:")
    print(
        df["six_class_intent"].value_counts()
    )

    texts = df["text"].astype(str)
    labels = df["six_class_intent"]

    all_results = []

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    print("\n" + "=" * 80)
    print("GENERATING 5-FOLD OOF PREDICTIONS")
    print("=" * 80)

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(texts, labels),
        start=1,
    ):

        print(
            f"\nFold {fold}/5..."
        )

        train_text = texts.iloc[train_idx]
        test_text = texts.iloc[test_idx]

        y_train = labels.iloc[train_idx]
        y_test = labels.iloc[test_idx]

        word_pred = train_word_model(
            train_text,
            test_text,
            y_train,
        )

        word_char_pred = train_word_char_model(
            train_text,
            test_text,
            y_train,
        )

        for i, actual, word_p, wc_p in zip(
            test_idx,
            y_test,
            word_pred,
            word_char_pred,
        ):

            all_results.append({
                "index": i,
                "actual": actual,
                "word_pred": word_p,
                "word_char_pred": wc_p,
            })

    results = pd.DataFrame(all_results)

    results["text"] = results["index"].apply(
        lambda i: df.iloc[i]["text"]
    )

    if "conversation_id" in df.columns:
        results["conversation_id"] = results[
            "index"
        ].apply(
            lambda i: df.iloc[i]["conversation_id"]
        )

    # ================================================================
    # OVERALL PERFORMANCE
    # ================================================================

    print("\n")
    print("=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)

    word_macro = f1_score(
        results["actual"],
        results["word_pred"],
        average="macro",
    )

    wc_macro = f1_score(
        results["actual"],
        results["word_char_pred"],
        average="macro",
    )

    print(
        f"\nWORD BIGRAMS Macro F1: "
        f"{word_macro:.4f}"
    )

    print(
        f"WORD + CHAR Macro F1: "
        f"{wc_macro:.4f}"
    )

    print(
        f"Change: "
        f"{wc_macro - word_macro:+.4f}"
    )

    # ================================================================
    # PER-CLASS ANALYSIS
    # ================================================================

    class_names = sorted(
        results["actual"].unique()
    )

    print("\n")
    print("=" * 80)
    print("PER-CLASS ERROR TRANSITIONS")
    print("=" * 80)

    summary = []

    for class_name in class_names:

        class_df = results[
            results["actual"] == class_name
        ]

        word_correct = (
            class_df["word_pred"]
             == class_df["actual"]
        )

        wc_correct = (
            class_df["word_char_pred"]
            == class_df["actual"]
        )

        corrected = (
            (~word_correct)
            & wc_correct
        ).sum()

        broken = (
            word_correct
            & (~wc_correct)
        ).sum()

        both_correct = (
            word_correct
            & wc_correct
        ).sum()

        both_wrong = (
            (~word_correct)
            & (~wc_correct)
        ).sum()

        print(
            class_name,
            "len =", len(class_df),
            "word_correct =", word_correct.sum(),
            "wc_correct =", wc_correct.sum(),
            "corrected =", corrected,
            "broken =", broken,
            "both_correct =", both_correct,
            "both_wrong =", both_wrong,
        )

        summary.append({
            "class": class_name,
            "total": len(class_df),
            "corrected": corrected,
            "broken": broken,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "net_change": corrected - broken,
        })

        print(
            f"\n{class_name}"
        )

        print(
            f"  Total examples:   {len(class_df)}"
        )

        print(
            f"  Corrected:        {corrected}"
        )

        print(
            f"  Broken:           {broken}"
        )

        print(
            f"  Both correct:     {both_correct}"
        )

        print(
            f"  Both wrong:       {both_wrong}"
        )

        print(
            f"  Net change:       "
            f"{corrected - broken:+d}"
        )

    # ================================================================
    # CONFUSION MATRICES
    # ================================================================

    print("\n")
    print("=" * 80)
    print("WORD BIGRAMS CONFUSION MATRIX")
    print("=" * 80)

    matrix_word = confusion_matrix(
        results["actual"],
        results["word_pred"],
        labels=class_names,
    )

    print(
        pd.DataFrame(
            matrix_word,
            index=class_names,
            columns=class_names,
        ).to_string()
    )

    print("\n")
    print("=" * 80)
    print("WORD + CHAR CONFUSION MATRIX")
    print("=" * 80)

    matrix_wc = confusion_matrix(
        results["actual"],
        results["word_char_pred"],
        labels=class_names,
    )

    print(
        pd.DataFrame(
            matrix_wc,
            index=class_names,
            columns=class_names,
        ).to_string()
    )

    # ================================================================
    # CONFUSION PAIR CHANGES
    # ================================================================

    print("\n")
    print("=" * 80)
    print("CONFUSION PAIR CHANGES")
    print("=" * 80)

    print(
        "\nPositive = WORD+CHAR made fewer errors."
    )

    print(
        "Negative = WORD+CHAR made more errors."
    )

    changes = []

    for i, actual in enumerate(class_names):

        for j, predicted in enumerate(class_names):

            if i == j:
                continue

            word_count = matrix_word[i, j]
            wc_count = matrix_wc[i, j]

            change = word_count - wc_count

            if change != 0:

                changes.append({
                    "actual": actual,
                    "predicted": predicted,
                    "word": word_count,
                    "word_char": wc_count,
                    "change": change,
                })

    changes.sort(
        key=lambda x: abs(x["change"]),
        reverse=True,
    )

    for item in changes:

        print(
            f"{item['actual']:30} -> "
            f"{item['predicted']:30} : "
            f"{item['word']} -> "
            f"{item['word_char']} "
            f"(change {item['change']:+d})"
        )

    # ================================================================
    # SAVE SUMMARY
    # ================================================================

    summary_df = pd.DataFrame(summary)

    output_path = (
        "data/analysis/"
        "word_vs_word_char_class_summary.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nSummary saved to: "
        f"{output_path}"
    )

    print(
        "\nNo production dataset was modified."
    )

    print(
        "No production training code was modified."
    )


if __name__ == "__main__":
    main()