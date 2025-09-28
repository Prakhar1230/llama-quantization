import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.gptq import GPTQQuantizer
import logging
from typing import Optional, Dict, Any
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading and preparing models for quantization."""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            self.config = {}
        
        # Force CPU usage - no CUDA
        self.device = "cpu"
        self.dtype = torch.float32  # Use float32 for CPU
        
        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            logger.info("CUDA not available. Using CPU-only mode.")
        
    def load_model_and_tokenizer(self, model_name: str) -> tuple:
        """Load model and tokenizer with CPU-only optimization."""
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with explicit CPU-only configuration
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=None,  # Don't use device_map for CPU-only
                offload_folder=None,  # No offloading needed for CPU
                offload_state_dict=False
            )
            
            # Explicitly move model to CPU
            model = model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Model parameters: {model.num_parameters():,}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def get_model_info(self, model) -> Dict[str, Any]:
        """Get model information for analysis."""
        return {
            "num_parameters": model.num_parameters(),
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
            "dtype": str(model.dtype),
            "device": str(next(model.parameters()).device)
        }
