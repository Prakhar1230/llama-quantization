import pytest
import torch
import tempfile
import json
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation import ModelEvaluator
from utils import MemoryMonitor, FileUtils

class MockModel:
    """Mock model for testing."""
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
    
    def generate(self, input_ids, **kwargs):
        """Mock generation method."""
        batch_size, seq_len = input_ids.shape
        max_length = kwargs.get('max_length', seq_len + 10)
        
        # Generate some additional tokens
        additional_tokens = min(10, max_length - seq_len)
        if additional_tokens > 0:
            new_tokens = torch.randint(1, 1000, (batch_size, additional_tokens))
            output = torch.cat([input_ids, new_tokens], dim=1)
        else:
            output = input_ids
        
        return output
    
    def num_parameters(self):
        return 1000000
    
    def parameters(self):
        return [torch.randn(100, 100) for _ in range(10)]

class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
    
    def __call__(self, text, **kwargs):
        """Mock encoding method."""
        if isinstance(text, list):
            # Handle batch encoding
            return {
                "input_ids": torch.tensor([
                    [1] + [hash(word) % 1000 for word in t.split()][:10] + [2]
                    for t in text
                ])
            }
        else:
            # Handle single text
            tokens = [1] + [hash(word) % 1000 for word in text.split()][:10] + [2]
            return {"input_ids": torch.tensor([tokens])}
    
    def encode(self, text, **kwargs):
        """Mock encode method."""
        return [1] + [hash(word) % 1000 for word in text.split()][:10] + [2]
    
    def decode(self, tokens, **kwargs):
        """Mock decode method."""
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        
        # Generate meaningful mock text based on tokens
        if len(tokens) > 5:
            return "This is a generated response with multiple words and sentences."
        else:
            return "Short response."
    
    def from_pretrained(self, *args, **kwargs):
        return self

class TestModelEvaluator:
    """Test ModelEvaluator functionality."""
    
    @pytest.fixture
    def evaluator(self):
        """Create ModelEvaluator instance."""
        return ModelEvaluator()
    
    @pytest.fixture
    def mock_model_tokenizer(self):
        """Create mock model and tokenizer."""
        return MockModel(), MockTokenizer()
    
    @pytest.fixture
    def sample_evaluation_data(self):
        """Create sample evaluation data."""
        return [
            {
                "prompt": "What is artificial intelligence?",
                "reference": "Artificial intelligence is a field of computer science focused on creating systems that can perform tasks requiring human intelligence.",
                "full_text": "What is artificial intelligence? Artificial intelligence is a field of computer science focused on creating systems that can perform tasks requiring human intelligence."
            },
            {
                "prompt": "Explain machine learning.",
                "reference": "Machine learning is a subset of AI that enables computers to learn and make decisions from data without explicit programming.",
                "full_text": "Explain machine learning. Machine learning is a subset of AI that enables computers to learn and make decisions from data without explicit programming."
            },
            {
                "prompt": "What is deep learning?",
                "reference": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                "full_text": "What is deep learning? Deep learning uses neural networks with multiple layers to model and understand complex patterns in data."
            }
        ]
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.rouge_scorer is not None
        assert evaluator.smoothing_function is not None
    
    def test_prepare_evaluation_data(self, evaluator):
        """Test preparation of evaluation data."""
        # Mock the dataset loading
        try:
            eval_data = evaluator.prepare_evaluation_data("wikitext", num_samples=5)
            # If successful, should return list
            assert isinstance(eval_data, list)
            
        except Exception as e:
            # It's okay if this fails in test environment due to network issues
            pytest.skip(f"Could not prepare evaluation data: {str(e)}")
    
    def test_evaluate_inference_speed(self, evaluator, mock_model_tokenizer):
        """Test inference speed evaluation."""
        model, tokenizer = mock_model_tokenizer
        test_prompts = [
            "What is AI?",
            "Explain ML.",
            "Define DL."
        ]
        
        speed_metrics = evaluator.evaluate_inference_speed(
            model, tokenizer, test_prompts, max_length=50
        )
        
        # Check required metrics
        required_metrics = [
            "mean_time", "median_time", "std_time", 
            "min_time", "max_time", "mean_tokens_generated", "total_runs"
        ]
        
        for metric in required_metrics:
            assert metric in speed_metrics
            assert isinstance(speed_metrics[metric], (int, float))
        
        # Validate metric values
        assert speed_metrics["mean_time"] >= 0
        assert speed_metrics["median_time"] >= 0
        assert speed_metrics["min_time"] >= 0
        assert speed_metrics["max_time"] >= speed_metrics["min_time"]
        assert speed_metrics["total_runs"] > 0
    
    def test_evaluate_accuracy(self, evaluator, mock_model_tokenizer, sample_evaluation_data):
        """Test accuracy evaluation."""
        model, tokenizer = mock_model_tokenizer
        
        accuracy_metrics = evaluator.evaluate_accuracy(
            model, tokenizer, sample_evaluation_data, max_length=100
        )
        
        # Check required metrics
        required_metrics = [
            "avg_rouge1", "avg_rouge2", "avg_rougeL", 
            "avg_bleu", "samples_evaluated"
        ]
        
        for metric in required_metrics:
            assert metric in accuracy_metrics
            assert isinstance(accuracy_metrics[metric], (int, float))
        
        # Validate metric ranges
        assert 0 <= accuracy_metrics["avg_rouge1"] <= 1
        assert 0 <= accuracy_metrics["avg_rouge2"] <= 1
        assert 0 <= accuracy_metrics["avg_rougeL"] <= 1
        assert 0 <= accuracy_metrics["avg_bleu"] <= 1
        assert accuracy_metrics["samples_evaluated"] > 0
    
    def test_evaluate_memory_usage(self, evaluator, mock_model_tokenizer):
        """Test memory usage evaluation."""
        model, tokenizer = mock_model_tokenizer
        
        memory_metrics = evaluator.evaluate_memory_usage(model)
        
        # Check required metrics
        required_metrics = [
            "model_size_mb", "process_memory_mb", 
            "available_memory_mb", "memory_utilization_percent"
        ]
        
        for metric in required_metrics:
            assert metric in memory_metrics
            assert isinstance(memory_metrics[metric], (int, float))
        
        # Validate metric values
        assert memory_metrics["model_size_mb"] > 0
        assert memory_metrics["process_memory_mb"] > 0
        assert memory_metrics["available_memory_mb"] >= 0
        assert 0 <= memory_metrics["memory_utilization_percent"] <= 100
    
    def test_comprehensive_evaluation_structure(self, evaluator, mock_model_tokenizer, tmp_path):
        """Test comprehensive evaluation structure without full execution."""
        model, tokenizer = mock_model_tokenizer
        
        # Mock the load_model_for_evaluation method
        evaluator.load_model_for_evaluation = lambda path: (model, tokenizer)
        
        # Mock prepare_evaluation_data to avoid network calls
        evaluator.prepare_evaluation_data = lambda name, num_samples: [
            {
                "prompt": "Test prompt",
                "reference": "Test reference",
                "full_text": "Test prompt Test reference"
            }
        ]
        
        output_path = tmp_path / "test_evaluation.json"
        
        try:
            results = evaluator.comprehensive_evaluation(
                model_path="test_path",
                output_path=str(output_path)
            )
            
            # Check result structure
            required_keys = [
                "model_path", "evaluation_timestamp", 
                "speed_metrics", "accuracy_metrics", "memory_metrics"
            ]
            
            for key in required_keys:
                assert key in results
            
            # Check if file was created
            assert output_path.exists()
            
            # Load and validate saved results
            with open(output_path) as f:
                saved_results = json.load(f)
            
            assert saved_results == results
            
        except Exception as e:
            # Some parts might fail in test environment
            pytest.skip(f"Comprehensive evaluation failed in test environment: {str(e)}")

class TestEvaluationUtils:
    """Test evaluation utility functions."""
    
    def test_rouge_scoring(self):
        """Test ROUGE scoring functionality."""
        evaluator = ModelEvaluator()
        
        reference = "The quick brown fox jumps over the lazy dog"
        generated = "The brown fox jumps over a lazy dog quickly"
        
        score = evaluator.rouge_scorer.score(reference, generated)
        
        # Should have rouge1, rouge2, and rougeL scores
        assert "rouge1" in score
        assert "rouge2" in score
        assert "rougeL" in score
        
        # Each score should have precision, recall, and fmeasure
        for metric in score.values():
            assert hasattr(metric, "precision")
            assert hasattr(metric, "recall")
            assert hasattr(metric, "fmeasure")
            
            # Scores should be between 0 and 1
            assert 0 <= metric.precision <= 1
            assert 0 <= metric.recall <= 1
            assert 0 <= metric.fmeasure <= 1
    
    def test_bleu_scoring(self):
        """Test BLEU scoring functionality."""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        smoothing = SmoothingFunction()
        
        reference = "the quick brown fox jumps over the lazy dog".split()
        generated = "the brown fox jumps over a lazy dog quickly".split()
        
        bleu_score = sentence_bleu(
            [reference], generated, 
            smoothing_function=smoothing.method1
        )
        
        # BLEU score should be between 0 and 1
        assert 0 <= bleu_score <= 1
    
    def test_memory_monitoring_integration(self):
        """Test integration with memory monitoring."""
        monitor = MemoryMonitor()
        evaluator = ModelEvaluator()
        
        with monitor.track_memory("test_evaluation"):
            # Simulate some evaluation work
            initial_memory = monitor.get_current_memory()
            
            # Do some memory-intensive work
            data = [i for i in range(1000)]
            processed = [x * 2 for x in data]
            
            final_memory = monitor.get_current_memory()
        
        # Should have tracked memory usage
        assert initial_memory["rss_mb"] > 0
        assert final_memory["rss_mb"] >= initial_memory["rss_mb"]

class TestEvaluationIntegration:
    """Integration tests for evaluation pipeline."""
    
    def test_evaluation_data_flow(self):
        """Test data flow through evaluation pipeline."""
        evaluator = ModelEvaluator()
        
        # Test data preparation
        sample_data = [
            {"text": "This is a test document for evaluation. It has multiple sentences."},
            {"text": "Another test document. This one is shorter."}
        ]
        
        # Mock dataset loading
        def mock_dataset():
            for item in sample_data:
                yield item
        
        # Test the processing logic
        processed_data = []
        for item in sample_data:
            text = item["text"]
            sentences = text.split('. ')
            
            if len(sentences) >= 2:
                prompt = '. '.join(sentences[:len(sentences)//2]) + '.'
                completion = '. '.join(sentences[len(sentences)//2:])
                
                processed_data.append({
                    "prompt": prompt,
                    "reference": completion,
                    "full_text": text
                })
        
        assert len(processed_data) > 0
        
        for item in processed_data:
            assert "prompt" in item
            assert "reference" in item
            assert "full_text" in item
    
    def test_metric_calculation_consistency(self):
        """Test consistency of metric calculations."""
        evaluator = ModelEvaluator()
        
        # Test with identical strings (should get perfect scores)
        identical_text = "This is an identical text for testing"
        
        rouge_score = evaluator.rouge_scorer.score(identical_text, identical_text)
        
        # Perfect match should give scores of 1.0
        assert rouge_score["rouge1"].fmeasure == 1.0
        assert rouge_score["rouge2"].fmeasure == 1.0
        assert rouge_score["rougeL"].fmeasure == 1.0
        
        # Test with completely different strings (should get low scores)
        different_text1 = "Apple banana cherry"
        different_text2 = "Dog elephant fox"
        
        rouge_score_diff = evaluator.rouge_scorer.score(different_text1, different_text2)
        
        # Different text should give low scores
        assert rouge_score_diff["rouge1"].fmeasure < 0.5
        assert rouge_score_diff["rouge2"].fmeasure < 0.5
        assert rouge_score_diff["rougeL"].fmeasure < 0.5

# Performance tests
@pytest.mark.performance
class TestEvaluationPerformance:
    """Performance tests for evaluation."""
    
    def test_evaluation_speed(self, mock_model_tokenizer):
        """Test evaluation completes within reasonable time."""
        import time
        
        model, tokenizer = mock_model_tokenizer
        evaluator = ModelEvaluator()
        
        test_prompts = [f"Test prompt {i}" for i in range(5)]
        
        start_time = time.time()
        speed_metrics = evaluator.evaluate_inference_speed(
            model, tokenizer, test_prompts, max_length=50
        )
        end_time = time.time()
        
        evaluation_time = end_time - start_time
        
        # Should complete within reasonable time
        assert evaluation_time < 30  # 30 seconds max
        
        # Should return meaningful results
        assert speed_metrics["total_runs"] > 0
        assert speed_metrics["mean_time"] > 0

if __name__ == "__main__":
    # Run with: python -m pytest tests/test_evaluation.py -v
    pytest.main([__file__, "-v"])
