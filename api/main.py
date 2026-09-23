from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from api.schemas import NewsRequest, PredictionResponse


MODEL_PATH = Path("models/linear_svm.pkl")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")


app = FastAPI(
    title="Fake News Detection API",
    description="API for classifying news statements as Fake or True.",
    version="1.0.0"
)


model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "linear_svm"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: NewsRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="News statement cannot be empty."
        )

    text_vectorized = vectorizer.transform([text])

    prediction = model.predict(text_vectorized)[0]

    probabilities = model.predict_proba(
        text_vectorized
    )[0]

    confidence = float(
        probabilities[prediction]
    )

    label = "True" if prediction == 1 else "Fake"

    return {
        "prediction": label,
        "confidence": confidence
    }