from fastapi.testclient import TestClient
from mlops_hatespeech.app import app


def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Those fucking democrats, they want our children to fail!"})
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
