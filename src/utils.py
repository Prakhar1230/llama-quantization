import os
import json
import yaml
import logging
import torch
import psutil
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manage configuration files and settings."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            raise
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """Save configuration to YAML file."""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {str(e)}")
            raise

class MemoryMonitor:
    """Monitor system memory usage during operations."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_current_memory()
    
    def get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = self.process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent_used": virtual_memory.percent,
            "available_mb": virtual_memory.available / (1024 * 1024),
            "total_mb": virtual_memory.total / (1024 * 1024)
        }
    
    def get_memory_diff(self) -> Dict[str, float]:
        """Get memory usage difference from initialization."""
        current = self.get_current_memory()
        return {
            "rss_diff_mb": current["rss_mb"] - self.initial_memory["rss_mb"],
            "vms_diff_mb": current["vms_mb"] - self.initial_memory["vms_mb"],
            "percent_diff": current["percent_used"] - self.initial_memory["percent_used"]
        }
    
    @contextmanager
    def track_memory(self, operation_name: str = "operation"):
        """Context manager to track memory usage during operations."""
        start_memory = self.get_current_memory()
        start_time = time.time()
        
        logger.info(f"Starting {operation_name} - Memory: {start_memory['rss_mb']:.2f}MB")
        
        try:
            yield self
        finally:
            end_memory = self.get_current_memory()
            end_time = time.time()
            
            memory_diff = end_memory["rss_mb"] - start_memory["rss_mb"]
            time_taken = end_time - start_time
            
            logger.info(f"Completed {operation_name} - Time: {time_taken:.2f}s, Memory diff: {memory_diff:.2f}MB")

class ModelUtils:
    """Utility functions for model operations."""
    
    @staticmethod
    def get_model_size(model) -> Dict[str, Any]:
        """Calculate model size and parameter count."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            "parameters": model.num_parameters(),
            "param_size_mb": param_size / (1024 * 1024),
            "buffer_size_mb": buffer_size / (1024 * 1024),
            "total_size_mb": total_size / (1024 * 1024),
            "total_size_gb": total_size / (1024 * 1024 * 1024)
        }
    
    @staticmethod
    def estimate_memory_requirement(model_size_gb: float, quantization_bits: int = 16) -> Dict[str, float]:
        """Estimate memory requirements for model operations."""
        # Base memory for model
        base_memory = model_size_gb * (quantization_bits / 16)
        
        # Additional memory for gradients (during training)
        gradient_memory = base_memory
        
        # Additional memory for optimizer states (Adam: 2x parameters)
        optimizer_memory = base_memory * 2
        
        # Additional memory for intermediate activations (rough estimate)
        activation_memory = base_memory * 0.5
        
        return {
            "model_memory_gb": base_memory,
            "gradient_memory_gb": gradient_memory,
            "optimizer_memory_gb": optimizer_memory,
            "activation_memory_gb": activation_memory,
            "total_training_gb": base_memory + gradient_memory + optimizer_memory + activation_memory,
            "total_inference_gb": base_memory + activation_memory
        }

class FileUtils:
    """Utility functions for file operations."""
    
    @staticmethod
    def ensure_dir_exists(directory: Union[str, Path]) -> Path:
        """Ensure directory exists, create if not."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """Save data to JSON file."""
        filepath = Path(filepath)
        FileUtils.ensure_dir_exists(filepath.parent)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Data saved to {filepath}")
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load data from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Data loaded from {filepath}")
        return data
    
    @staticmethod
    def get_file_size(filepath: Union[str, Path]) -> Dict[str, float]:
        """Get file size in different units."""
        size_bytes = Path(filepath).stat().st_size
        
        return {
            "bytes": size_bytes,
            "kb": size_bytes / 1024,
            "mb": size_bytes / (1024 * 1024),
            "gb": size_bytes / (1024 * 1024 * 1024)
        }

class TextProcessor:
    """Text processing utilities for model evaluation."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        import re
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        return text.strip()
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 512) -> str:
        """Truncate text to maximum length while preserving word boundaries."""
        if len(text) <= max_length:
            return text
        
        # Find last space before max_length
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > 0:
            return truncated[:last_space] + '...'
        else:
            return truncated + '...'
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

class QuantizationUtils:
    """Utilities specific to quantization operations."""
    
    @staticmethod
    def calculate_compression_ratio(original_size: float, quantized_size: float) -> Dict[str, float]:
        """Calculate compression metrics."""
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        size_reduction = original_size - quantized_size
        size_reduction_percent = (size_reduction / original_size) * 100 if original_size > 0 else 0
        
        return {
            "compression_ratio": compression_ratio,
            "size_reduction_mb": size_reduction,
            "size_reduction_percent": size_reduction_percent
        }
    
    @staticmethod
    def estimate_quantization_speedup(bits_original: int = 16, bits_quantized: int = 4) -> Dict[str, float]:
        """Estimate theoretical speedup from quantization."""
        # Simplified speedup estimation based on bit reduction
        memory_speedup = bits_original / bits_quantized
        compute_speedup = (bits_original / bits_quantized) ** 0.5  # Square root approximation
        
        return {
            "memory_speedup": memory_speedup,
            "compute_speedup": compute_speedup,
            "theoretical_speedup": (memory_speedup + compute_speedup) / 2
        }
    
    @staticmethod
    def validate_quantization_config(config: Dict[str, Any]) -> bool:
        """Validate quantization configuration."""
        required_fields = ["bits", "group_size", "model_seqlen"]
        
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field in quantization config: {field}")
                return False
        
        # Validate bit precision
        if config["bits"] not in [2, 4, 8, 16]:
            logger.error(f"Invalid bit precision: {config['bits']}. Supported: [2, 4, 8, 16]")
            return False
        
        # Validate group size
        if config["group_size"] <= 0:
            logger.error(f"Invalid group size: {config['group_size']}. Must be positive.")
            return False
        
        return True

class BenchmarkUtils:
    """Utilities for benchmarking and performance measurement."""
    
    @staticmethod
    def measure_inference_time(model, tokenizer, prompts: List[str], 
                             max_length: int = 128, num_runs: int = 3) -> Dict[str, Any]:
        """Measure inference time statistics."""
        times = []
        tokens_generated = []
        
        for prompt in prompts[:min(len(prompts), 10)]:  # Limit for CPU benchmarking
            run_times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                inference_time = time.time() - start_time
                run_times.append(inference_time)
                
                # Count tokens generated
                input_tokens = inputs["input_ids"].shape[1]
                output_tokens = outputs.shape[1] - input_tokens
                tokens_generated.append(output_tokens)
            
            times.extend(run_times)
        
        return {
            "mean_time": np.mean(times),
            "median_time": np.median(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "mean_tokens_generated": np.mean(tokens_generated) if tokens_generated else 0,
            "total_runs": len(times)
        }

# Global utility functions
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up logging configuration."""
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def check_system_requirements() -> Dict[str, Any]:
    """Check if system meets requirements for quantization."""
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    requirements_met = {
        "sufficient_memory": memory.total >= 6 * 1024 * 1024 * 1024,  # 6GB minimum
        "sufficient_cpu": cpu_count >= 2,
        "pytorch_available": True,
        "transformers_available": True
    }
    
    try:
        import torch
        import transformers
    except ImportError as e:
        requirements_met["pytorch_available"] = False
        requirements_met["transformers_available"] = False
        logger.error(f"Missing required packages: {str(e)}")
    
    system_info = {
        "total_memory_gb": memory.total / (1024 * 1024 * 1024),
        "available_memory_gb": memory.available / (1024 * 1024 * 1024),
        "cpu_count": cpu_count,
        "requirements_met": all(requirements_met.values()),
        "checks": requirements_met
    }
