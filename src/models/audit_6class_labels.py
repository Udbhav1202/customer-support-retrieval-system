import pandas as pd
from collections import Counter

CSV_PATH = "data/processed/merged_training_dataset_cleaned.csv"

TEXT_COL = "text"
LABEL_COL = "intent"

TARGET_CLASSES = [
    "CUSTOMER_SUPPORT",
    "OTHER",
    "SWITCHING_RETENTION",
    "NETWORK_CONNECTIVITY",
    "BILLING_PAYMENT",
    "WIFI_ROUTER",
]

RARE_CLASSES = {
    "FRAUD_SECURITY",
    "MOBILE_PHONE_SERVICE",
    "REFUND",
    "SIM_CARD",
}


def print_separator():
    print("\n" + "=" * 80)


def main():
    print("=" * 80)
    print("6-CLASS LABEL QUALITY AUDIT")
    print("=" * 80)

    df = pd.read_csv(CSV_PATH)

    print(f"\nOriginal dataset: {df.shape}")

    # Remove pseudo-labeled rows if the dataset contains that column.
    if "is_pseudo_labeled" in df.columns:
        before = len(df)

        df = df[
            df["is_pseudo_labeled"].fillna(False).astype(bool) == False
        ].copy()

        print(f"Excluded pseudo-labeled rows: {before - len(df)}")

    # Keep only the six classes we're currently evaluating.
    df = df[df[LABEL_COL].isin(TARGET_CLASSES)].copy()

    print(f"6-class audit dataset: {df.shape}")

    print("\nClass distribution:")
    print(df[LABEL_COL].value_counts())

    # ------------------------------------------------------------------
    # 1. BASIC LABEL AUDIT
    # ------------------------------------------------------------------

    print_separator()
    print("1. BASIC LABEL AUDIT")
    print_separator()

    for label in TARGET_CLASSES:
        subset = df[df[LABEL_COL] == label]

        print(f"\n{'CLASS: ' + label}")
        print(f"Examples: {len(subset)}")

        for _, row in subset.head(10).iterrows():
            text = str(row[TEXT_COL]).replace("\n", " ")

            print(f"  - {text}")

    # ------------------------------------------------------------------
    # 2. CUSTOMER_SUPPORT AUDIT
    # ------------------------------------------------------------------

    print_separator()
    print("2. CUSTOMER_SUPPORT AUDIT")
    print_separator()

    subset = df[df[LABEL_COL] == "CUSTOMER_SUPPORT"]

    print(f"Total CUSTOMER_SUPPORT examples: {len(subset)}")

    print("\nFirst 30 examples:")

    for i, (_, row) in enumerate(subset.head(30).iterrows(), start=1):
        print(f"\n[{i}]")
        print(str(row[TEXT_COL]).replace("\n", " "))

    # ------------------------------------------------------------------
    # 3. OTHER AUDIT
    # ------------------------------------------------------------------

    print_separator()
    print("3. OTHER AUDIT")
    print_separator()

    subset = df[df[LABEL_COL] == "OTHER"]

    print(f"Total OTHER examples: {len(subset)}")

    print("\nALL OTHER examples:")

    for i, (_, row) in enumerate(subset.iterrows(), start=1):
        print(f"\n[{i}]")
        print(str(row[TEXT_COL]).replace("\n", " "))

    # ------------------------------------------------------------------
    # 4. SWITCHING_RETENTION AUDIT
    # ------------------------------------------------------------------

    print_separator()
    print("4. SWITCHING_RETENTION AUDIT")
    print_separator()

    subset = df[df[LABEL_COL] == "SWITCHING_RETENTION"]

    print(f"Total SWITCHING_RETENTION examples: {len(subset)}")

    for i, (_, row) in enumerate(subset.head(30).iterrows(), start=1):
        print(f"\n[{i}]")
        print(str(row[TEXT_COL]).replace("\n", " "))

    # ------------------------------------------------------------------
    # 5. NETWORK_CONNECTIVITY AUDIT
    # ------------------------------------------------------------------

    print_separator()
    print("5. NETWORK_CONNECTIVITY AUDIT")
    print_separator()

    subset = df[df[LABEL_COL] == "NETWORK_CONNECTIVITY"]

    print(f"Total NETWORK_CONNECTIVITY examples: {len(subset)}")

    for i, (_, row) in enumerate(subset.head(30).iterrows(), start=1):
        print(f"\n[{i}]")
        print(str(row[TEXT_COL]).replace("\n", " "))

    # ------------------------------------------------------------------
    # 6. BILLING_PAYMENT AUDIT
    # ------------------------------------------------------------------

    print_separator()
    print("6. BILLING_PAYMENT AUDIT")
    print_separator()

    subset = df[df[LABEL_COL] == "BILLING_PAYMENT"]

    print(f"Total BILLING_PAYMENT examples: {len(subset)}")

    for i, (_, row) in enumerate(subset.head(30).iterrows(), start=1):
        print(f"\n[{i}]")
        print(str(row[TEXT_COL]).replace("\n", " "))

    # ------------------------------------------------------------------
    # 7. WIFI_ROUTER AUDIT
    # ------------------------------------------------------------------

    print_separator()
    print("7. WIFI_ROUTER AUDIT")
    print_separator()

    subset = df[df[LABEL_COL] == "WIFI_ROUTER"]

    print(f"Total WIFI_ROUTER examples: {len(subset)}")

    for i, (_, row) in enumerate(subset.head(30).iterrows(), start=1):
        print(f"\n[{i}]")
        print(str(row[TEXT_COL]).replace("\n", " "))

    # ------------------------------------------------------------------
    # 8. DUPLICATE / NEAR-DUPLICATE TEXT CHECK
    # ------------------------------------------------------------------

    print_separator()
    print("8. DUPLICATE TEXT CHECK")
    print_separator()

    duplicated = df[df[TEXT_COL].duplicated(keep=False)].sort_values(TEXT_COL)

    print(f"Rows involved in exact duplicate texts: {len(duplicated)}")

    if len(duplicated) > 0:
        for text, group in duplicated.groupby(TEXT_COL):
            labels = list(group[LABEL_COL])

            if len(set(labels)) > 1:
                print("\nPOTENTIAL LABEL CONFLICT:")
                print(f"Text: {text}")
                print(f"Labels: {labels}")

    # ------------------------------------------------------------------
    # 9. SAME CONVERSATION ID WITH DIFFERENT LABELS
    # ------------------------------------------------------------------

    print_separator()
    print("9. CONVERSATION ID LABEL CONFLICT CHECK")
    print_separator()

    if "conversation_id" in df.columns:
        conflicts = (
            df.groupby("conversation_id")[LABEL_COL]
            .nunique()
        )

        conflicts = conflicts[conflicts > 1]

        print(
            f"Conversation IDs with multiple labels: {len(conflicts)}"
        )

        if len(conflicts) > 0:
            print("\nPotential conflicts:")

            for conversation_id in conflicts.index:
                rows = df[df["conversation_id"] == conversation_id]

                print(f"\nConversation ID: {conversation_id}")

                for _, row in rows.iterrows():
                    print(
                        f"  [{row[LABEL_COL]}] "
                        f"{str(row[TEXT_COL]).replace(chr(10), ' ')}"
                    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    print_separator()
    print("AUDIT COMPLETE")
    print_separator()

    print(
        "\nIMPORTANT:"
        "\nThis script does NOT modify the dataset."
        "\nIt only prints examples for manual label inspection."
    )


if __name__ == "__main__":
    main()