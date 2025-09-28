#!/usr/bin/env python3

import argparse
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name: str, save_path: str):
    """Download and save model locally."""
    logger.info(f"Downloading model: {model_name}")
    
    try:
        # Create directory
        os.makedirs(save_path, exist_ok=True)
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(save_path)
        logger.info("Tokenizer downloaded successfully")
        
        # Download model with CPU-friendly settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.save_pretrained(save_path)
        logger.info(f"Model downloaded and saved to {save_path}")
        
        # Save model info
        model_info = {
            "model_name": model_name,
            "num_parameters": model.num_parameters(),
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        }
        
        with open(os.path.join(save_path, "model_info.yaml"), 'w') as f:
            yaml.dump(model_info, f, default_flow_style=False)
        
        logger.info(f"Model info: {model_info}")
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LLM model")
    parser.add_argument("--model_name", required=True, help="HuggingFace model name")
    parser.add_argument("--save_path", default="models/original", help="Path to save model")
    
    args = parser.parse_args()
    download_model(args.model_name, args.save_path)
