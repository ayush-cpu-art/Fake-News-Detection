from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = Path("data/processed/cleaned_data.csv")


def load_data():
    """Load the cleaned dataset."""
    return pd.read_csv(DATA_PATH)


def create_tfidf_features(df):
    """Convert news statements into TF-IDF features."""

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(df["Statement"])
    y = df["Label"]

    return X, y, vectorizer


if __name__ == "__main__":
    df = load_data()

    X, y, vectorizer = create_tfidf_features(df)

    print("Dataset shape:", df.shape)
    print("TF-IDF matrix shape:", X.shape)
    print("Number of features:", len(vectorizer.get_feature_names_out()))