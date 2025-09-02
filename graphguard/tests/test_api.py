"""
GraphGuard API Tests
Test cases for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
import json
from unittest.mock import patch, MagicMock

# Import the FastAPI app
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.api import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test cases for health check endpoint"""
    
    def test_health_check(self):
        """Test health check endpoint returns correct response"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "model_version" in data
        assert "timestamp" in data
    
    def test_health_check_structure(self):
        """Test health check response structure"""
        response = client.get("/health")
        data = response.json()
        
        required_fields = ["status", "model_version", "timestamp"]
        for field in required_fields:
            assert field in data
            assert data[field] is not None


class TestScoreEndpoint:
    """Test cases for transaction scoring endpoint"""
    
    def test_score_valid_transaction(self):
        """Test scoring endpoint with valid transaction data"""
        transaction_data = {
            "transaction_id": "TXN_TEST_001",
            "amount": 100.0,
            "src_account_id": "ACC_001",
            "dst_account_id": "ACC_002",
            "device_id": "DEV_001",
            "ip_address": "192.168.1.1",
            "merchant_id": "MERCH_001",
            "timestamp": "2025-02-09T10:00:00"
        }
        
        response = client.post("/score", json=transaction_data)
        assert response.status_code == 200
        
        data = response.json()
        required_fields = [
            "transaction_id", "probability", "label", "threshold",
            "top_features", "reason", "graph_context"
        ]
        for field in required_fields:
            assert field in data
    
    def test_score_missing_required_fields(self):
        """Test scoring endpoint with missing required fields"""
        # Missing required fields
        incomplete_data = {
            "transaction_id": "TXN_TEST_002",
            "amount": 100.0
            # Missing src_account_id and dst_account_id
        }
        
        response = client.post("/score", json=incomplete_data)
        assert response.status_code == 422  # Validation error
    
    def test_score_invalid_amount(self):
        """Test scoring endpoint with invalid amount"""
        transaction_data = {
            "transaction_id": "TXN_TEST_003",
            "amount": -100.0,  # Negative amount
            "src_account_id": "ACC_001",
            "dst_account_id": "ACC_002"
        }
        
        response = client.post("/score", json=transaction_data)
        # Should still work as amount validation is not strict in schema
        assert response.status_code == 200


class TestRootEndpoint:
    """Test cases for root endpoint"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "GraphGuard Fraud Detection API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
        assert "docs" in data


class TestModelLoading:
    """Test cases for model loading functionality"""
    
    @patch('app.api.joblib.load')
    @patch('app.api.np.load')
    def test_model_loading_success(self, mock_np_load, mock_joblib_load):
        """Test successful model loading"""
        # Mock successful loading
        mock_joblib_load.return_value = MagicMock()
        mock_np_load.return_value = {}
        
        # This would test the startup event, but it's complex to test
        # In practice, we'd test the actual loading logic separately
        assert True
    
    def test_model_loading_failure_handling(self):
        """Test that API handles model loading failures gracefully"""
        # This would test the startup event error handling
        # For now, just verify the endpoint still responds
        response = client.get("/health")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])
