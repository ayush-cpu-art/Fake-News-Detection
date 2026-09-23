import re
from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/IFND.csv")
PROCESSED_DATA_PATH = Path("data/processed/cleaned_data.csv")


def clean_text(text):
    """Clean a news statement for NLP processing."""
    text = str(text)

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_data():
    """Load the raw IFND dataset."""
    return pd.read_csv(
        RAW_DATA_PATH,
        encoding="latin-1"
    )


def remove_conflicting_statements(df):
    """Remove statements that have both TRUE and Fake labels."""

    label_counts = (
        df.groupby("Statement")["Label"]
        .nunique()
    )

    conflicting_statements = label_counts[
        label_counts > 1
    ].index

    df = df[
        ~df["Statement"].isin(conflicting_statements)
    ].copy()

    return df


def preprocess_data(df):
    """Clean and prepare the dataset."""

    # Remove conflicting statements
    df = remove_conflicting_statements(df)

    # Remove duplicate statements
    df = df.drop_duplicates(
        subset="Statement",
        keep="first"
    ).copy()

    # Clean text
    df["Statement"] = df["Statement"].apply(clean_text)

    # Convert labels
    df["Label"] = (
        df["Label"]
        .str.strip()
        .str.lower()
        .map({
            "true": 1,
            "fake": 0
        })
    )

    # Keep only the columns required for text classification
    df = df[["Statement", "Label"]]

    # Remove any rows that became invalid
    df = df.dropna().copy()

    return df


def save_processed_data(df):
    """Save the processed dataset."""
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )


def main():
    print("Loading dataset...")

    df = load_data()

    print(f"Original dataset: {len(df)} rows")

    df = preprocess_data(df)

    print(f"Processed dataset: {len(df)} rows")

    print("Label distribution:")
    print(df["Label"].value_counts())

    save_processed_data(df)

    print(
        f"Processed dataset saved to: "
        f"{PROCESSED_DATA_PATH}"
    )


if __name__ == "__main__":
    main()