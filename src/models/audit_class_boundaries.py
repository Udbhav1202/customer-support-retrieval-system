import pandas as pd

CSV_PATH = "data/processed/merged_training_dataset_cleaned.csv"

TARGET_COL = "intent"
TEXT_COL = "text"

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


def print_section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def inspect_class(df, class_name, top_n=15):
    class_df = df[
        df["six_class_intent"] == class_name
    ].copy()

    print_section(
        f"{class_name} ({len(class_df)} examples)"
    )

    print("\nOriginal intent distribution:")
    print(class_df["intent"].value_counts())

    print("\nExamples:")

    for _, row in class_df.head(top_n).iterrows():
        print("\n" + "-" * 80)

        print(f"Conversation ID: {row['conversation_id']}")
        print(f"Original intent: {row['intent']}")

        print("\nText:")
        print(row[TEXT_COL])

def main():

    print_section(
        "6-CLASS BOUNDARY AUDIT"
    )

    df = pd.read_csv(CSV_PATH)

    print(
        f"\nOriginal dataset: {df.shape}"
    )

    # Remove pseudo-labeled rows if present.
    if "is_pseudo_labeled" in df.columns:

        df = df[
            df["is_pseudo_labeled"] != True
        ].copy()

    # Remove rare classes.
    df = df[
        ~df[TARGET_COL].isin(RARE_CLASSES)
    ].copy()

    # Build the same 6-class taxonomy
    # used in our previous experiments.
    df["six_class_intent"] = (
        df[TARGET_COL]
        .apply(apply_6class_taxonomy)
    )

    print(
        f"Evaluation dataset: {df.shape}"
    )

    print_section(
        "6-CLASS DISTRIBUTION"
    )

    print(
        df["six_class_intent"]
        .value_counts()
    )

    # --------------------------------------------------------------
    # Focus only on the classes currently causing major confusion.
    # --------------------------------------------------------------

    inspect_class(
        df,
        "CUSTOMER_SUPPORT"
    )

    inspect_class(
        df,
        "SWITCHING_RETENTION"
    )

    inspect_class(
        df,
        "OTHER"
    )

    print_section(
        "BOUNDARY AUDIT COMPLETE"
    )

    print(
        "\nNo files were modified."
    )


if __name__ == "__main__":
    main()