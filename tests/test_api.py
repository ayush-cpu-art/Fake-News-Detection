from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "linear_svm"


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "text": (
                "NASA has announced the discovery of a new planet "
                "in our solar system that is larger than Jupiter."
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in ["Fake", "True"]
    assert 0 <= data["confidence"] <= 1


def test_empty_prediction():
    response = client.post(
        "/predict",
        json={
            "text": ""
        }
    )

    assert response.status_code == 422