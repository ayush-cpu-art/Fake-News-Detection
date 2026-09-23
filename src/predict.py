import joblib
from pathlib import Path


MODEL_PATH = Path("models/linear_svm.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")


def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def predict_news(text):
    model, vectorizer = load_model()

    text_vectorized = vectorizer.transform([text])

    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]

    confidence = probabilities[prediction]

    label = "True" if prediction == 1 else "Fake"

    return label, confidence


if __name__ == "__main__":

    news = input("Enter a news statement: ")

    prediction, confidence = predict_news(news)

    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")