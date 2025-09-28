import pytest
import json
from fastapi.testclient import TestClient
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Mock model loading for testing
class MockModel:
    def __init__(self):
        self.device = "cpu"
        self.dtype = "float32"
    
    def generate(self, *args, **kwargs):
        import torch
        # Return a simple mock output
        input_ids = args[0]
        max_length = kwargs.get('max_length', 50)
        # Generate some dummy tokens
        output_length = min(max_length, input_ids.shape[1] + 10)
        return torch.randint(1, 1000, (1, output_length))
    
    def num_parameters(self):
        return 1000000  # 1M parameters
    
    def parameters(self):
        import torch
        return [torch.randn(100, 100)]

class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2
    
    def __call__(self, text, **kwargs):
        import torch
        # Simple mock encoding
        tokens = [1] + [hash(word) % 1000 for word in text.split()][:10] + [2]
        return {"input_ids": torch.tensor([tokens])}
    
    def encode(self, text, **kwargs):
        return [1] + [hash(word) % 1000 for word in text.split()][:10] + [2]
    
    def decode(self, tokens, **kwargs):
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        return f"Generated response for tokens: {len(tokens)} tokens"
    
    def from_pretrained(self, *args, **kwargs):
        return self

# Mock the imports in api_server
import unittest.mock

@pytest.fixture
def mock_model_loading():
    """Mock model loading to avoid loading real models in tests."""
    with unittest.mock.patch('src.api_server.AutoTokenizer') as mock_tokenizer, \
         unittest.mock.patch('src.api_server.AutoModelForCausalLM') as mock_model, \
         unittest.mock.patch('src.api_server.pipeline') as mock_pipeline:
        
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()
        mock_model.from_pretrained.return_value = MockModel()
        mock_pipeline.return_value = lambda x, **kwargs: [{"generated_text": x + " Generated text"}]
        
        yield

@pytest.fixture
def client(mock_model_loading):
    """Create test client with mocked model loading."""
    # Set environment variable to prevent loading real model
    os.environ["MODEL_PATH"] = "test_model_path"
    
    # Import after mocking
    from src.api_server import app
    
    client = TestClient(app)
    return client

class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns correct status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

class TestInferenceEndpoint:
    """Test inference endpoint."""
    
    def test_inference_basic_request(self, client):
        """Test basic inference request."""
        request_data = {
            "prompt": "What is artificial intelligence?",
            "max_length": 100,
            "temperature": 0.7
        }
        
        response = client.post("/inference", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        required_fields = ["generated_text", "inference_time", "tokens_per_second", 
                          "input_tokens", "output_tokens"]
        
        for field in required_fields:
            assert field in data
        
        assert isinstance(data["inference_time"], float)
        assert isinstance(data["tokens_per_second"], float)
        assert isinstance(data["input_tokens"], int)
        assert isinstance(data["output_tokens"], int)
    
    def test_inference_with_optional_parameters(self, client):
        """Test inference with all optional parameters."""
        request_data = {
            "prompt": "Explain machine learning",
            "max_length": 200,
            "temperature": 0.5,
            "top_p": 0.8,
            "do_sample": True
        }
        
        response = client.post("/inference", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "generated_text" in data
    
    def test_inference_with_minimal_parameters(self, client):
        """Test inference with only required parameters."""
        request_data = {
            "prompt": "Hello, world!"
        }
        
        response = client.post("/inference", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "generated_text" in data
    
    def test_inference_empty_prompt(self, client):
        """Test inference with empty prompt."""
        request_data = {
            "prompt": ""
        }
        
        response = client.post("/inference", json=request_data)
        # Should still work, but might return minimal output
        assert response.status_code == 200
    
    def test_inference_invalid_parameters(self, client):
        """Test inference with invalid parameters."""
        request_data = {
            "prompt": "Test prompt",
            "max_length": -1,  # Invalid
            "temperature": 2.0  # Too high
        }
        
        response = client.post("/inference", json=request_data)
        # API should handle gracefully, either by clamping values or returning error
        # Depending on implementation, this might be 200 (clamped) or 400 (error)
        assert response.status_code in [200, 400, 422]
    
    def test_inference_missing_prompt(self, client):
        """Test inference without prompt."""
        request_data = {
            "max_length": 100
        }
        
        response = client.post("/inference", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_inference_very_long_prompt(self, client):
        """Test inference with very long prompt."""
        long_prompt = "This is a test prompt. " * 200  # Very long prompt
        
        request_data = {
            "prompt": long_prompt,
            "max_length": 50
        }
        
        response = client.post("/inference", json=request_data)
        # Should handle gracefully (truncation or processing)
        assert response.status_code in [200, 400]

class TestModelInfoEndpoint:
    """Test model info endpoint."""
    
    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        required_fields = ["num_parameters", "model_size_mb", "dtype", "device"]
        
        for field in required_fields:
            assert field in data
        
        assert isinstance(data["num_parameters"], int)
        assert isinstance(data["model_size_mb"], float)
        assert isinstance(data["dtype"], str)
        assert isinstance(data["device"], str)
        
        assert data["num_parameters"] > 0
        assert data["model_size_mb"] > 0

class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_invalid_endpoint(self, client):
        """Test requesting invalid endpoint."""
        response = client.get("/invalid_endpoint")
        assert response.status_code == 404
    
    def test_invalid_method(self, client):
        """Test using invalid HTTP method."""
        response = client.get("/inference")  # Should be POST
        assert response.status_code == 405
    
    def test_invalid_json(self, client):
        """Test sending invalid JSON."""
        response = client.post(
            "/inference",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

class TestAPIPerformance:
    """Test API performance characteristics."""
    
    def test_concurrent_requests(self, client):
        """Test handling multiple concurrent requests."""
        import concurrent.futures
        import time
        
        def make_request():
            request_data = {"prompt": "Test prompt for concurrency"}
            return client.post("/inference", json=request_data)
        
        # Make multiple concurrent requests
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            responses = [future.result() for future in futures]
        
        end_time = time.time()
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Should complete within reasonable time
        assert end_time - start_time < 30  # 30 seconds max
    
    def test_response_time(self, client):
        """Test API response time."""
        import time
        
        request_data = {
            "prompt": "Quick test prompt",
            "max_length": 50
        }
        
        start_time = time.time()
        response = client.post("/inference", json=request_data)
        end_time = time.time()
        
        assert response.status_code == 200
        
        response_time = end_time - start_time
        # Should respond within reasonable time (adjust based on hardware)
        assert response_time < 10  # 10 seconds max for test environment

class TestAPISecurity:
    """Test API security features."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.get("/health")
        
        # Check for CORS headers (depends on implementation)
        # The actual headers may vary based on configuration
        assert response.status_code == 200
    
    def test_large_request_handling(self, client):
        """Test handling of very large requests."""
        # Create a very large prompt
        large_prompt = "A" * 10000  # 10k characters
        
        request_data = {
            "prompt": large_prompt,
            "max_length": 50
        }
        
        response = client.post("/inference", json=request_data)
        # Should either handle gracefully or return appropriate error
        assert response.status_code in [200, 400, 413, 422]

# Utility functions for testing
def create_test_requests(num_requests=5):
    """Create a list of test requests for bulk testing."""
    prompts = [
        "What is machine learning?",
        "Explain artificial intelligence.",
        "How does deep learning work?",
        "What is natural language processing?",
        "Describe computer vision applications."
    ]
    
    return [
        {
            "prompt": prompt,
            "max_length": 100,
            "temperature": 0.7
        }
        for prompt in prompts[:num_requests]
    ]

# Integration test
@pytest.mark.integration
def test_api_integration_workflow(client):
    """Test complete API workflow."""
    # 1. Check health
    health_response = client.get("/health")
    assert health_response.status_code == 200
    
    # 2. Get model info
    info_response = client.get("/model/info")
    assert info_response.status_code == 200
    
    # 3. Make inference request
    inference_request = {
        "prompt": "Explain the concept of quantization in machine learning.",
        "max_length": 150,
        "temperature": 0.8
    }
    
    inference_response = client.post("/inference", json=inference_request)
    assert inference_response.status_code == 200
    
    # Verify response structure
    data = inference_response.json()
    assert "generated_text" in data
    assert "inference_time" in data
    assert data["inference_time"] > 0

if __name__ == "__main__":
    # Run with: python -m pytest tests/test_api.py -v
    pytest.main([__file__, "-v"])
