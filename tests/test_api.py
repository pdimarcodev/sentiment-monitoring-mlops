import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from src.sentiment_analyzer.api import app

class TestSentimentAPI:
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "sentiment-analysis"
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_info" in data
        assert "timestamp" in data
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_type" in data
        assert "labels" in data
    
    def test_predict_endpoint(self, client):
        """Test single prediction endpoint"""
        payload = {"text": "I love this product!"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "confidence" in data
        assert "text" in data
        assert "all_scores" in data
        assert data["sentiment"] in ["positive", "negative", "neutral"]
    
    def test_predict_batch_endpoint(self, client):
        """Test batch prediction endpoint"""
        payload = {
            "texts": [
                "I love this!",
                "This is terrible!",
                "This is okay."
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_processed" in data
        assert len(data["results"]) == 3
        assert data["total_processed"] == 3
    
    def test_predict_empty_text_error(self, client):
        """Test error handling for empty text"""
        payload = {"text": ""}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Pydantic validation error
    
    def test_predict_invalid_payload(self, client):
        """Test error handling for invalid payload"""
        payload = {"invalid_field": "test"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Metrics should be in Prometheus format
        assert "sentiment_" in response.text
    
    def test_batch_size_limit(self, client):
        """Test batch size limit"""
        # Create payload with more than 100 texts
        large_payload = {"texts": ["test text"] * 101}
        response = client.post("/predict/batch", json=large_payload)
        assert response.status_code == 422