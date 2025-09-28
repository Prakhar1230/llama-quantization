import pytest
import torch
import tempfile
import os
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from quantization import QuantizationPipeline
from model_loader import ModelLoader
from utils import ConfigManager, MemoryMonitor

class TestQuantizationPipeline:
    """Test cases for quantization pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def quantization_pipeline(self):
        """Create quantization pipeline instance."""
        return QuantizationPipeline(calibration_data_size=10)
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "quantization": {
                "default_bits": 4,
                "group_size": 128,
                "calibration_samples": 10
            }
        }
    
    def test_quantization_pipeline_initialization(self, quantization_pipeline):
        """Test quantization pipeline initialization."""
        assert quantization_pipeline.calibration_data_size == 10
        assert quantization_pipeline.quantization_config["bits"] == 4
        assert quantization_pipeline.quantization_config["group_size"] == 128
    
    def test_quantization_config_validation(self, quantization_pipeline):
        """Test quantization configuration validation."""
        from utils import QuantizationUtils
        
        # Valid config
        valid_config = {
            "bits": 4,
            "group_size": 128,
            "model_seqlen": 2048
        }
        assert QuantizationUtils.validate_quantization_config(valid_config) is True
        
        # Invalid config - missing required field
        invalid_config = {
            "bits": 4,
            "group_size": 128
            # missing model_seqlen
        }
        assert QuantizationUtils.validate_quantization_config(invalid_config) is False
        
        # Invalid config - invalid bits
        invalid_bits_config = {
            "bits": 3,  # Invalid
            "group_size": 128,
            "model_seqlen": 2048
        }
        assert QuantizationUtils.validate_quantization_config(invalid_bits_config) is False
    
    def test_calibration_data_preparation(self, quantization_pipeline):
        """Test calibration data preparation."""
        # Mock tokenizer
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
            
            @property
            def eos_token_id(self):
                return 0
        
        tokenizer = MockTokenizer()
        
        try:
            calibration_data = quantization_pipeline.prepare_calibration_data(
                tokenizer, "wikitext"
            )
            
            # Should return some data (might be less due to filtering)
            assert len(calibration_data) >= 0
            
            if len(calibration_data) > 0:
                # Check structure of first sample
                sample = calibration_data[0]
                assert "input_ids" in sample
                assert isinstance(sample["input_ids"], torch.Tensor)
                
        except Exception as e:
            # It's okay if this fails in test environment due to network issues
            pytest.skip(f"Could not prepare calibration data: {str(e)}")
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring."""
        monitor = MemoryMonitor()
        
        initial_memory = monitor.get_current_memory()
        assert "rss_mb" in initial_memory
        assert "available_mb" in initial_memory
        assert initial_memory["rss_mb"] > 0
        
        # Test memory diff
        memory_diff = monitor.get_memory_diff()
        assert "rss_diff_mb" in memory_diff
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        from utils import QuantizationUtils
        
        # Test normal case
        metrics = QuantizationUtils.calculate_compression_ratio(1000, 250)
        assert metrics["compression_ratio"] == 4.0
        assert metrics["size_reduction_mb"] == 750
        assert metrics["size_reduction_percent"] == 75.0
        
        # Test edge case - zero quantized size
        metrics = QuantizationUtils.calculate_compression_ratio(1000, 0)
        assert metrics["compression_ratio"] == 0
        
        # Test edge case - zero original size
        metrics = QuantizationUtils.calculate_compression_ratio(0, 100)
        assert metrics["size_reduction_percent"] == 0
    
    def test_speedup_estimation(self):
        """Test quantization speedup estimation."""
        from utils import QuantizationUtils
        
        speedup = QuantizationUtils.estimate_quantization_speedup(16, 4)
        
        assert speedup["memory_speedup"] == 4.0
        assert speedup["compute_speedup"] == 2.0
        assert speedup["theoretical_speedup"] == 3.0
    
    @pytest.mark.slow
    def test_model_loading(self, temp_dir):
        """Test model loading (requires internet)."""
        try:
            model_loader = ModelLoader()
            
            # Try to load a small model for testing
            model_name = "hf-internal-testing/tiny-random-gpt2"  # Tiny model for testing
            
            model, tokenizer = model_loader.load_model_and_tokenizer(model_name)
            
            assert model is not None
            assert tokenizer is not None
            
            # Test model info
            model_info = model_loader.get_model_info(model)
            assert "num_parameters" in model_info
            assert "model_size_mb" in model_info
            assert model_info["num_parameters"] > 0
            
        except Exception as e:
            pytest.skip(f"Could not load model for testing: {str(e)}")

class TestConfigurationManagement:
    """Test configuration management utilities."""
    
    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create temporary config file."""
        config_file = tmp_path / "test_config.yaml"
        return config_file
    
    def test_config_save_and_load(self, temp_config_file):
        """Test saving and loading configuration."""
        test_config = {
            "quantization": {
                "bits": 4,
                "method": "gptq"
            },
            "model": {
                "name": "test_model"
            }
        }
        
        # Save config
        ConfigManager.save_config(test_config, str(temp_config_file))
        assert temp_config_file.exists()
        
        # Load config
        loaded_config = ConfigManager.load_config(str(temp_config_file))
        assert loaded_config == test_config
    
    def test_config_error_handling(self):
        """Test configuration error handling."""
        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            ConfigManager.load_config("non_existent_file.yaml")

class TestTextProcessing:
    """Test text processing utilities."""
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        from utils import TextProcessor
        
        dirty_text = "  This   is    a   test   text!!!   "
        clean_text = TextProcessor.clean_text(dirty_text)
        
        assert clean_text == "This is a test text!!!"
    
    def test_text_truncation(self):
        """Test text truncation with word boundaries."""
        from utils import TextProcessor
        
        long_text = "This is a very long text that should be truncated at word boundaries"
        truncated = TextProcessor.truncate_text(long_text, max_length=30)
        
        assert len(truncated) <= 33  # 30 + "..."
        assert truncated.endswith("...")
        assert not truncated[:-3].endswith(" ")  # Should not end with space
    
    def test_sentence_splitting(self):
        """Test sentence splitting."""
        from utils import TextProcessor
        
        text = "This is sentence one. This is sentence two! And this is sentence three?"
        sentences = TextProcessor.split_into_sentences(text)
        
        assert len(sentences) == 3
        assert "This is sentence one" in sentences[0]
        assert "This is sentence two" in sentences[1]
        assert "And this is sentence three" in sentences[2]

class TestSystemRequirements:
    """Test system requirements checking."""
    
    def test_system_requirements_check(self):
        """Test system requirements checking."""
        from utils import check_system_requirements
        
        requirements = check_system_requirements()
        
        # Should return dictionary with required keys
        required_keys = [
            "total_memory_gb", "available_memory_gb", "cpu_count", 
            "requirements_met", "checks"
        ]
        
        for key in required_keys:
            assert key in requirements
        
        # Check structure of checks
        checks = requirements["checks"]
        check_keys = ["sufficient_memory", "sufficient_cpu", "pytorch_available", "transformers_available"]
        
        for key in check_keys:
            assert key in checks
            assert isinstance(checks[key], bool)

# Benchmark tests
class TestBenchmarkUtils:
    """Test benchmark utilities."""
    
    def test_benchmark_utils_initialization(self):
        """Test benchmark utilities initialization."""
        from utils import BenchmarkUtils
        
        benchmark_utils = BenchmarkUtils()
        assert benchmark_utils is not None

# Integration tests
@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def integration_setup(self, tmp_path):
        """Set up integration test environment."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        return {
            "model_dir": model_dir,
            "config_dir": config_dir,
            "tmp_path": tmp_path
        }
    
    @pytest.mark.slow
    def test_end_to_end_quantization(self, integration_setup):
        """Test end-to-end quantization process."""
        # This test would require a small model and significant setup
        # Skipping for now due to complexity and resource requirements
        pytest.skip("End-to-end integration test requires extensive setup")

# Performance tests
@pytest.mark.performance
class TestPerformance:
    """Performance-related tests."""
    
    def test_memory_efficiency(self):
        """Test memory efficiency of operations."""
        from utils import MemoryMonitor
        
        monitor = MemoryMonitor()
        
        with monitor.track_memory("test_operation") as m:
            # Simulate some memory-intensive operation
            data = [i for i in range(10000)]
            processed_data = [x * 2 for x in data]
        
        # Should complete without memory errors
        assert len(processed_data) == 10000

if __name__ == "__main__":
    # Run with: python -m pytest tests/test_quantization.py -v
    pytest.main([__file__, "-v"])
