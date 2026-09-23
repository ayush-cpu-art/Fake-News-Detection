import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


DATA_PATH = Path("data/processed/cleaned_data.csv")
MODEL_DIR = Path("models")


def load_data():
    return pd.read_csv(DATA_PATH)


def main():
    df = load_data()

    X_text = df["Statement"]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF training shape: {X_train_tfidf.shape}")
    print(f"TF-IDF testing shape: {X_test_tfidf.shape}")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "naive_bayes": MultinomialNB(),
        "linear_svm": CalibratedClassifierCV(
            LinearSVC(random_state=42),
            method="sigmoid",
            cv=3
        )
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for name, model in models.items():

        print(f"\n{'=' * 50}")
        print(f"Training: {name}")
        print(f"{'=' * 50}")

        start_time = time.time()

        model.fit(X_train_tfidf, y_train)

        training_time = time.time() - start_time

        predictions = model.predict(X_test_tfidf)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        results[name] = accuracy

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")

        print(
            classification_report(
                y_test,
                predictions,
                target_names=["Fake", "True"]
            )
        )

        model_path = MODEL_DIR / f"{name}.pkl"

        joblib.dump(
            model,
            model_path
        )

        print(f"Saved model: {model_path}")

    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"

    joblib.dump(
        vectorizer,
        vectorizer_path
    )

    print(f"\nSaved vectorizer: {vectorizer_path}")

    print("\nModel accuracy summary:")

    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.4f}")


if __name__ == "__main__":
    main()