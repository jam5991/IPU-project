from fastapi.testclient import TestClient
from deployment.app import app

client = TestClient(app)

def test_predict_endpoint():
    # Test data
    test_text = "I love this product!"
    response = client.post("/predict", json={"text": test_text})
    
    # Check the response status code
    assert response.status_code == 200, "Expected status code 200"
    
    # Check the structure of the response
    response_data = response.json()
    assert "sentiment" in response_data, "Response body should have a sentiment field"
    assert "probabilities" in response_data, "Response body should have a probabilities field"
    assert "positive" in response_data["probabilities"], "Probabilities should include positive"
    assert "negative" in response_data["probabilities"], "Probabilities should include negative"

def test_predict_endpoint_no_text():
    # Test case with missing text field
    response = client.post("/predict", json={})
    # Check for 422 Unprocessable Entity
    assert response.status_code == 422, "Expected status code 422 for invalid input"
