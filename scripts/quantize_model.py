#!/usr/bin/env python3

import argparse
import os
import sys
import logging
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_loader import ModelLoader
from quantization import QuantizationPipeline
from utils import MemoryMonitor, FileUtils, ConfigManager
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_system_compatibility():
    """Check system compatibility and warn about limitations."""
    logger.info("Checking system compatibility...")
    
    # Check PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        logger.warning("CUDA not available. Using CPU-only mode.")
        logger.warning("CPU quantization may be slower and limited in methods.")
    
    # Check available memory
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    logger.info(f"Available system memory: {available_memory_gb:.1f} GB")
    
    if available_memory_gb < 3:
        logger.warning("Low system memory detected. Quantization may fail or be very slow.")
        logger.warning("Consider closing other applications or using a smaller model.")
    
    return cuda_available

def main():
    parser = argparse.ArgumentParser(description="Quantize LLM model for CPU deployment")
    parser.add_argument("--input_path", required=True, help="Path to original model")
    parser.add_argument("--output_path", required=True, help="Path to save quantized model")
    parser.add_argument("--quantization_method", default="dynamic", 
                       choices=["dynamic"], help="Quantization method (only dynamic supported for CPU)")
    parser.add_argument("--config", default="configs/quantization_config.yaml",
                       help="Quantization configuration file")
    parser.add_argument("--bits", type=int, default=8, help="Quantization bits (8 for dynamic)")
    parser.add_argument("--calibration_samples", type=int, default=32,
                       help="Number of calibration samples (reduced for CPU)")
    parser.add_argument("--test_model", action="store_true", 
                       help="Test the quantized model after creation")
    
    args = parser.parse_args()
    
    # Check system compatibility
    cuda_available = check_system_compatibility()
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor()
    
    try:
        logger.info("Starting model quantization process...")
        start_time = time.time()
        
        # Load configuration
        if os.path.exists(args.config):
            config = ConfigManager.load_config(args.config)
        else:
            logger.warning(f"Config file {args.config} not found. Using defaults.")
            config = {
                "quantization": {
                    "default_bits": args.bits,
                    "calibration_samples": args.calibration_samples
                }
            }
        
        with memory_monitor.track_memory("model_loading"):
            # Load model and tokenizer
            model_loader = ModelLoader()
            
            # Check if input path contains model files or is a model name
            if os.path.isdir(args.input_path) and any(
                f.endswith(('.bin', '.safetensors', '.json')) 
                for f in os.listdir(args.input_path)
            ):
                # Local model directory
                logger.info("Loading model from local directory...")
                model, tokenizer = model_loader.load_model_and_tokenizer(args.input_path)
            else:
                logger.error(f"Model directory {args.input_path} not found or invalid.")
                logger.info("Please ensure the model was downloaded first using download_model.py")
                sys.exit(1)
            
            # Get model info
            model_info = model_loader.get_model_info(model)
            logger.info(f"Loaded model info: {model_info}")
        
        with memory_monitor.track_memory("quantization"):
            # Initialize quantization pipeline
            quantization_pipeline = QuantizationPipeline(
                calibration_data_size=args.calibration_samples
            )
            
            # Perform quantization - only dynamic method supported
            logger.info("Performing dynamic quantization...")
            quantization_pipeline.quantize_model_dynamic(
                model, tokenizer, args.output_path
            )
        
        # Test the quantized model if requested
        if args.test_model:
            try:
                from quantization import QuantizedModelLoader
                loader = QuantizedModelLoader()
                quantized_model, quantized_tokenizer = loader.load_quantized_model(args.output_path)
                
                if quantized_model and quantized_tokenizer:
                    logger.info("Quantized model loaded successfully!")
                    
                    # Quick test
                    test_input = "Hello, how are you?"
                    inputs = quantized_tokenizer(test_input, return_tensors="pt")
                    
                    with torch.no_grad():
                        outputs = quantized_model(**inputs)
                    
                    logger.info("Quantized model test passed!")
                else:
                    logger.warning("Could not test quantized model")
                    
            except Exception as e:
                logger.warning(f"Model testing failed: {str(e)}")
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        final_memory = memory_monitor.get_current_memory()
        memory_diff = memory_monitor.get_memory_diff()
        
        # Get file sizes for comparison
        try:
            original_files = [f for f in os.listdir(args.input_path) 
                            if f.endswith(('.bin', '.safetensors'))]
            original_size = sum(
                FileUtils.get_file_size(os.path.join(args.input_path, f))["mb"]
                for f in original_files
            ) if original_files else model_info.get('model_size_mb', 0)
            
            quantized_files = [f for f in os.listdir(args.output_path) 
                             if f.endswith(('.pt', '.pkl', '.bin'))]
            quantized_size = sum(
                FileUtils.get_file_size(os.path.join(args.output_path, f))["mb"]
                for f in quantized_files
            ) if quantized_files else 0
            
            if quantized_size > 0:
                compression_ratio = original_size / quantized_size
                size_reduction_percent = ((original_size - quantized_size) / original_size) * 100
            else:
                compression_ratio = 0
                size_reduction_percent = 0
                
        except Exception as e:
            logger.warning(f"Could not calculate file size metrics: {str(e)}")
            compression_ratio = 0
            size_reduction_percent = 0
            original_size = 0
            quantized_size = 0
        
        # Save quantization report
        report = {
            "quantization_method": args.quantization_method,
            "quantization_bits": args.bits,
            "calibration_samples": args.calibration_samples,
            "original_model_path": args.input_path,
            "quantized_model_path": args.output_path,
            "original_model_info": model_info,
            "file_sizes": {
                "original_mb": original_size,
                "quantized_mb": quantized_size,
                "compression_ratio": compression_ratio,
                "size_reduction_percent": size_reduction_percent
            },
            "performance_metrics": {
                "total_time_seconds": total_time,
                "memory_usage": final_memory,
                "memory_diff": memory_diff
            },
            "system_info": {
                "cuda_available": cuda_available,
                "pytorch_version": torch.__version__,
                "device_used": "cpu"
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        report_path = os.path.join(args.output_path, "quantization_report.json")
        FileUtils.save_json(report, report_path)
        
        logger.info("="*60)
        logger.info("QUANTIZATION COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Quantized model saved to: {args.output_path}")
        if compression_ratio > 0:
            logger.info(f"Original size: {original_size:.1f} MB")
            logger.info(f"Quantized size: {quantized_size:.1f} MB") 
            logger.info(f"Compression ratio: {compression_ratio:.2f}x")
            logger.info(f"Size reduction: {size_reduction_percent:.1f}%")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Peak memory usage: {final_memory['rss_mb']:.1f} MB")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error during quantization: {str(e)}")
        logger.error("This may be due to:")
        logger.error("1. Insufficient system memory (you have only 2.3GB available)")
        logger.error("2. Model too large for your system")
        logger.error("3. PyTorch version compatibility issues")
        logger.error("\nSuggestions:")
        logger.error("- Close other applications to free memory")
        logger.error("- Try a smaller model (e.g., microsoft/DialoGPT-small)")
        logger.error("- Restart your computer to free up memory")
        sys.exit(1)

if __name__ == "__main__":
    main()
