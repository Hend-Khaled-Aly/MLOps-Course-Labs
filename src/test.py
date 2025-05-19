from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json().get("message", "")

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    # Status should be either "healthy" or "model not loaded"
    assert response.json().get("status") in ["healthy", "model not loaded"]

def test_predict():
    payload = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_response = response.json()
    assert "churn_prediction" in json_response
    assert isinstance(json_response["churn_prediction"], int)
