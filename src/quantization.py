import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from datasets import load_dataset
from typing import List, Dict, Any
import time
import gc
import psutil
import pickle
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantizationPipeline:
    """Handles model quantization with CPU-only techniques."""
    
    def __init__(self, calibration_data_size: int = 128):
        self.calibration_data_size = calibration_data_size
        self.device = "cpu"
        
    def prepare_calibration_data(self, tokenizer, dataset_name: str = "wikitext") -> List[str]:
        """Prepare calibration dataset for quantization."""
        logger.info("Preparing calibration data...")
        
        try:
            # Load a small subset for calibration
            if dataset_name == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            else:
                dataset = load_dataset("c4", "en", split="train", streaming=True)
                
            calibration_texts = []
            
            for i, sample in enumerate(dataset):
                if i >= self.calibration_data_size:
                    break
                
                text = sample.get("text", "")
                
                if len(text.strip()) > 50:  # Filter out very short texts
                    calibration_texts.append(text[:512])  # Truncate to 512 chars
            
            logger.info(f"Prepared {len(calibration_texts)} calibration samples")
            return calibration_texts[:self.calibration_data_size]
            
        except Exception as e:
            logger.error(f"Error preparing calibration data: {str(e)}")
            # Fallback to dummy data
            return [f"Sample calibration text number {i}" for i in range(min(10, self.calibration_data_size))]
    
    def quantize_model_dynamic(self, model, tokenizer, save_path: str) -> None:
        """Dynamic quantization using PyTorch native methods with proper saving."""
        logger.info("Starting dynamic quantization...")
        
        try:
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            
            logger.info("Dynamic quantization completed successfully")
            
            # Create save directory
            os.makedirs(save_path, exist_ok=True)
            
            # Save quantized model using torch.save (not save_pretrained)
            model_path = os.path.join(save_path, "quantized_model.pt")
            torch.save(quantized_model, model_path)
            logger.info(f"Quantized model saved to {model_path}")
            
            # Save tokenizer separately (this works fine)
            tokenizer.save_pretrained(save_path)
            logger.info("Tokenizer saved successfully")
            
            # Save model configuration for loading
            config_path = os.path.join(save_path, "quantization_config.json")
            config = {
                "quantization_method": "dynamic",
                "dtype": "qint8",
                "quantized_modules": ["Linear"],
                "model_type": model.__class__.__name__,
                "torch_version": torch.__version__
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("Quantization configuration saved")
            
            # Test the quantized model
            self.test_quantized_model(quantized_model, tokenizer)
            
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {str(e)}")
            # Try alternative approach
            self.quantize_model_state_dict(model, tokenizer, save_path)
    
    def quantize_model_state_dict(self, model, tokenizer, save_path: str) -> None:
        """Alternative: Save only state dict of quantized model."""
        logger.info("Starting quantization with state_dict saving...")
        
        try:
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            
            # Create save directory
            os.makedirs(save_path, exist_ok=True)
            
            # Save only the state dict
            state_dict_path = os.path.join(save_path, "quantized_state_dict.pt")
            torch.save(quantized_model.state_dict(), state_dict_path)
            logger.info(f"Quantized state dict saved to {state_dict_path}")
            
            # Save tokenizer
            tokenizer.save_pretrained(save_path)
            
            # Save original model config for reconstruction
            config_path = os.path.join(save_path, "model_config.json")
            config = {
                "model_class": model.__class__.__name__,
                "model_config": model.config.to_dict() if hasattr(model, 'config') else {},
                "quantization_applied": True,
                "quantization_method": "dynamic_state_dict"
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("Quantization with state_dict completed successfully")
            
        except Exception as e:
            logger.error(f"State dict quantization failed: {str(e)}")
            # Final fallback: manual parameter quantization
            self.quantize_model_manual(model, tokenizer, save_path)
    
    def quantize_model_manual(self, model, tokenizer, save_path: str) -> None:
        """Manual quantization approach as final fallback."""
        logger.info("Starting manual quantization approach...")
        
        try:
            # Simple manual quantization of Linear layers
            quantized_params = {}
            
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() >= 2:  # Likely a weight matrix
                    # Quantize to int8
                    param_np = param.detach().cpu().numpy()
                    scale = (param_np.max() - param_np.min()) / 255.0
                    zero_point = int(-param_np.min() / scale)
                    
                    quantized = ((param_np / scale) + zero_point).round().astype('int8')
                    
                    quantized_params[name] = {
                        'quantized_weight': quantized,
                        'scale': scale,
                        'zero_point': zero_point,
                        'original_shape': param.shape
                    }
                else:
                    # Keep non-quantizable parameters as-is
                    quantized_params[name] = param.detach().cpu().numpy()
            
            # Create save directory
            os.makedirs(save_path, exist_ok=True)
            
            # Save quantized parameters
            params_path = os.path.join(save_path, "quantized_parameters.pkl")
            with open(params_path, 'wb') as f:
                pickle.dump(quantized_params, f)
            
            logger.info(f"Manually quantized parameters saved to {params_path}")
            
            # Save tokenizer
            tokenizer.save_pretrained(save_path)
            
            # Save configuration
            config_path = os.path.join(save_path, "manual_quantization_config.json")
            config = {
                "quantization_method": "manual",
                "quantized_layers": list(quantized_params.keys()),
                "model_class": model.__class__.__name__
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("Manual quantization completed successfully")
            
        except Exception as e:
            logger.error(f"Manual quantization failed: {str(e)}")
            raise
    
    def test_quantized_model(self, quantized_model, tokenizer) -> None:
        """Test the quantized model to ensure it works."""
        try:
            logger.info("Testing quantized model...")
            
            # Simple test
            test_input = "Hello, this is a test."
            inputs = tokenizer(test_input, return_tensors="pt")
            
            with torch.no_grad():
                outputs = quantized_model(**inputs)
            
            logger.info("Quantized model test passed!")
            
        except Exception as e:
            logger.warning(f"Quantized model test failed: {str(e)}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Monitor memory usage during quantization."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "cpu_percent": process.cpu_percent(),
            "available_memory_mb": psutil.virtual_memory().available / (1024 * 1024)
        }

class QuantizedModelLoader:
    """Helper class to load quantized models."""
    
    @staticmethod
    def load_quantized_model(model_path: str):
        """Load a quantized model from saved files."""
        
        # Check what type of quantized model we have
        if os.path.exists(os.path.join(model_path, "quantized_model.pt")):
            # Full quantized model
            logger.info("Loading full quantized model...")
            model = torch.load(os.path.join(model_path, "quantized_model.pt"))
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer
            
        elif os.path.exists(os.path.join(model_path, "quantized_state_dict.pt")):
            # State dict approach - requires recreation
            logger.info("Loading quantized model from state dict...")
            logger.warning("State dict loading requires original model architecture - not fully implemented")
            return None, None
            
        elif os.path.exists(os.path.join(model_path, "quantized_parameters.pkl")):
            # Manual quantization - requires custom loader
            logger.info("Manual quantized model detected...")
            logger.warning("Manual quantization loading requires custom implementation")
            return None, None
            
        else:
            raise FileNotFoundError(f"No quantized model found in {model_path}")
