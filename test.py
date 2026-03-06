from fastapi.testclient import TestClient
from app import app

# Test initialize
client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Marketing AI Running"}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_discount():
    payload = {
        "category": "Electronics",
        "actual_price": 1500.0,
        "rating": 4.2,
        "rating_count": 1250
    }
    response = client.post("/predict_discount", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_discount_percent" in data
    assert isinstance(data["predicted_discount_percent"], float)

def test_answer_question_single():
    payload = {
        "query": "What is the most expensive product?"
    }
    response = client.post("/answer_question", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)

def test_answer_question_batch():
    payload = {
        "queries": ["average price", "average rating"]
    }
    response = client.post("/answer_question", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 2
    assert "query" in data["results"][0]
    assert "answer" in data["results"][0]