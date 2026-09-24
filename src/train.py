import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


DATA_PATH = Path("data/processed/cleaned_data.csv")
MODEL_DIR = Path("models")


def load_data():
    return pd.read_csv(DATA_PATH)


def main():
    # MLflow configuration
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fake-news-detection")

    # Load dataset
    df = load_data()

    X_text = df["Statement"]
    y = df["Label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # TF-IDF feature engineering
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF training shape: {X_train_tfidf.shape}")
    print(f"TF-IDF testing shape: {X_test_tfidf.shape}")

    # Models
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "naive_bayes": MultinomialNB(),
        "linear_svm": CalibratedClassifierCV(
            LinearSVC(random_state=42),
            method="sigmoid",
            cv=3,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # Train models
    for name, model in models.items():

        print(f"\n{'=' * 50}")
        print(f"Training: {name}")
        print(f"{'=' * 50}")

        with mlflow.start_run(run_name=name):

            start_time = time.time()

            model.fit(X_train_tfidf, y_train)

            training_time = time.time() - start_time

            predictions = model.predict(X_test_tfidf)

            # Metrics
            accuracy = accuracy_score(y_test, predictions)

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
            )

            results[name] = accuracy

            # Log parameters
            mlflow.log_param("model", name)
            mlflow.log_param("max_features", 10000)
            mlflow.log_param("ngram_range", "(1, 2)")
            mlflow.log_param("min_df", 2)
            mlflow.log_param("max_df", 0.95)
            mlflow.log_param("stop_words", "english")
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision_weighted", precision)
            mlflow.log_metric("recall_weighted", recall)
            mlflow.log_metric("f1_weighted", f1)
            mlflow.log_metric(
                "training_time_seconds",
                training_time,
            )

            # Print results
            print(f"Training time: {training_time:.2f} seconds")
            print(f"Accuracy: {accuracy:.4f}")

            print(
                classification_report(
                    y_test,
                    predictions,
                    target_names=["Fake", "True"],
                )
            )

            # Save model
            model_path = MODEL_DIR / f"{name}.pkl"

            joblib.dump(
                model,
                model_path,
            )

            print(f"Saved model: {model_path}")

            # Log model file to MLflow
            mlflow.log_artifact(
                str(model_path),
                artifact_path="model",
            )

            print(f"MLflow run recorded: {name}")

    # Save TF-IDF vectorizer
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"

    joblib.dump(
        vectorizer,
        vectorizer_path,
    )

    print(f"\nSaved vectorizer: {vectorizer_path}")

    # Print model summary
    print("\nModel accuracy summary:")

    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.4f}")

    print("\nMLflow experiment: fake-news-detection")
    print("MLflow database: mlflow.db")


if __name__ == "__main__":
    main()