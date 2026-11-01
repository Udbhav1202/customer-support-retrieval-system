import pandas as pd
from pathlib import Path

CSV_PATH = "data/processed/merged_training_dataset_cleaned.csv"
OUTPUT_PATH = "data/analysis/6class_dataset_audit.csv"

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


def main():

    print_section("6-CLASS DATASET QUALITY AUDIT")

    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------

    df = pd.read_csv(CSV_PATH)

    print(f"\nOriginal dataset: {df.shape}")

    # Remove pseudo-labeled rows if present.
    if "is_pseudo_labeled" in df.columns:
        before = len(df)

        df = df[
            df["is_pseudo_labeled"] != True
        ].copy()

        print(
            f"Removed pseudo-labeled rows: "
            f"{before - len(df)}"
        )

    # Remove rare classes exactly as previous experiments.
    before = len(df)

    df = df[
        ~df["intent"].isin(RARE_CLASSES)
    ].copy()

    print(
        f"Removed rare-class rows: "
        f"{before - len(df)}"
    )

    # Create experimental taxonomy.
    df["six_class_intent"] = df["intent"].apply(
        apply_6class_taxonomy
    )

    print(f"\nEvaluation dataset: {df.shape}")

    # ------------------------------------------------------------------
    # CLASS DISTRIBUTION
    # ------------------------------------------------------------------

    print_section("6-CLASS DISTRIBUTION")

    distribution = (
        df["six_class_intent"]
        .value_counts()
    )

    print(distribution)

    # ------------------------------------------------------------------
    # TEXT CLEANING
    # ------------------------------------------------------------------

    df["text_clean"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["text_length"] = (
        df["text_clean"]
        .str.len()
    )

    df["word_count"] = (
        df["text_clean"]
        .str.split()
        .str.len()
    )

    # ------------------------------------------------------------------
    # AUDIT 1: VERY SHORT TEXTS
    # ------------------------------------------------------------------

    print_section("AUDIT 1: VERY SHORT TEXTS")

    short_df = df[
        df["word_count"] <= 5
    ].copy()

    print(
        f"\nVery short examples: "
        f"{len(short_df)}"
    )

    print(
        "\nDistribution by class:"
    )

    print(
        short_df["six_class_intent"]
        .value_counts()
    )

    # ------------------------------------------------------------------
    # AUDIT 2: DUPLICATES
    # ------------------------------------------------------------------

    print_section("AUDIT 2: DUPLICATE TEXTS")

    duplicate_mask = (
        df["text_clean"]
        .duplicated(
            keep=False
        )
    )

    duplicate_df = df[
        duplicate_mask
    ].copy()

    print(
        f"\nRows involved in duplicate texts: "
        f"{len(duplicate_df)}"
    )

    print(
        f"Unique duplicated texts: "
        f"{duplicate_df['text_clean'].nunique()}"
    )

    # ------------------------------------------------------------------
    # AUDIT 3: OTHER
    # ------------------------------------------------------------------

    print_section("AUDIT 3: OTHER")

    other_df = df[
        df["six_class_intent"] == "OTHER"
    ].copy()

    print(
        f"\nOTHER examples: "
        f"{len(other_df)}"
    )

    print(
        "\nShortest OTHER examples:"
    )

    print(
        other_df[
            [
                "conversation_id",
                "intent",
                "word_count",
                "text_clean",
            ]
        ]
        .sort_values("word_count")
        .head(15)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # AUDIT 4: NETWORK CONNECTIVITY
    # ------------------------------------------------------------------

    print_section(
        "AUDIT 4: NETWORK_CONNECTIVITY"
    )

    network_df = df[
        df["six_class_intent"]
        == "NETWORK_CONNECTIVITY"
    ].copy()

    print(
        f"\nNETWORK_CONNECTIVITY examples: "
        f"{len(network_df)}"
    )

    print(
        "\nOriginal-label distribution:"
    )

    print(
        network_df["intent"]
        .value_counts()
    )

    print(
        "\nShortest NETWORK examples:"
    )

    print(
        network_df[
            [
                "conversation_id",
                "intent",
                "word_count",
                "text_clean",
            ]
        ]
        .sort_values("word_count")
        .head(15)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # AUDIT 5: CUSTOMER SUPPORT
    # ------------------------------------------------------------------

    print_section(
        "AUDIT 5: CUSTOMER_SUPPORT"
    )

    support_df = df[
        df["six_class_intent"]
        == "CUSTOMER_SUPPORT"
    ].copy()

    print(
        f"\nCUSTOMER_SUPPORT examples: "
        f"{len(support_df)}"
    )

    print(
        "\nOriginal-label distribution:"
    )

    print(
        support_df["intent"]
        .value_counts()
    )

    print(
        "\nShortest CUSTOMER_SUPPORT examples:"
    )

    print(
        support_df[
            [
                "conversation_id",
                "intent",
                "word_count",
                "text_clean",
            ]
        ]
        .sort_values("word_count")
        .head(15)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # BUILD AUDIT DATASET
    # ------------------------------------------------------------------

    print_section(
        "BUILDING AUDIT CSV"
    )

    audit_frames = []

    # Very short texts
    if len(short_df) > 0:

        temp = short_df.copy()

        temp["audit_reason"] = (
            "VERY_SHORT_TEXT"
        )

        audit_frames.append(temp)

    # Duplicate texts
    if len(duplicate_df) > 0:

        temp = duplicate_df.copy()

        temp["audit_reason"] = (
            "DUPLICATE_TEXT"
        )

        audit_frames.append(temp)

    # OTHER
    if len(other_df) > 0:

        temp = other_df.copy()

        temp["audit_reason"] = (
            "OTHER_CLASS"
        )

        audit_frames.append(temp)

    # Combine
    if audit_frames:

        audit_df = pd.concat(
            audit_frames,
            ignore_index=True
        )

        # Remove exact duplicate rows
        audit_df = audit_df.drop_duplicates(
            subset=[
                "conversation_id",
                "text_clean",
                "audit_reason",
            ]
        )

    else:

        audit_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    output_path = Path(
        OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    audit_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nAudit rows saved: "
        f"{len(audit_df)}"
    )

    print(
        f"Output file: "
        f"{OUTPUT_PATH}"
    )

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------

    print_section(
        "AUDIT COMPLETE"
    )

    print(
        "\nOriginal dataset was NOT modified."
    )

    print(
        "Only an analysis CSV was created."
    )


if __name__ == "__main__":
    main()
