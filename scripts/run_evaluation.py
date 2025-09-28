#!/usr/bin/env python3

import argparse
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation import ModelEvaluator
from utils import MemoryMonitor, FileUtils, ConfigManager
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized LLM model")
    parser.add_argument("--model_path", required=True, help="Path to quantized model")
    parser.add_argument("--output_path", default="results/evaluation.json",
                       help="Path to save evaluation results")
    parser.add_argument("--dataset", default="wikitext",
                       choices=["wikitext", "c4"], help="Evaluation dataset")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of samples to evaluate")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum generation length")
    parser.add_argument("--config", default="configs/model_config.yaml",
                       help="Model configuration file")
    
    args = parser.parse_args()
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor()
    
    try:
        logger.info("Starting model evaluation...")
        start_time = time.time()
        
        # Load configuration if available
        if os.path.exists(args.config):
            config = ConfigManager.load_config(args.config)
            logger.info("Configuration loaded")
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        with memory_monitor.track_memory("comprehensive_evaluation"):
            # Run comprehensive evaluation
            results = evaluator.comprehensive_evaluation(
                model_path=args.model_path,
                output_path=args.output_path
            )
        
        # Add additional metadata
        total_time = time.time() - start_time
        final_memory = memory_monitor.get_current_memory()
        
        results.update({
            "evaluation_config": {
                "dataset": args.dataset,
                "num_samples": args.num_samples,
                "max_length": args.max_length,
                "total_evaluation_time": total_time
            },
            "system_info": {
                "final_memory_usage": final_memory,
                "peak_memory_diff": memory_monitor.get_memory_diff()
            }
        })
        
        # Save updated results
        FileUtils.save_json(results, args.output_path)
        
        # Print summary
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {args.output_path}")
        
        # Print key metrics
        speed_metrics = results.get("speed_metrics", {})
        accuracy_metrics = results.get("accuracy_metrics", {})
        memory_metrics = results.get("memory_metrics", {})
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if speed_metrics:
            print(f"Average inference time: {speed_metrics.get('avg_inference_time', 0):.3f}s")
            print(f"Tokens per second: {speed_metrics.get('avg_tokens_per_second', 0):.2f}")
        
        if accuracy_metrics:
            print(f"Average ROUGE-1: {accuracy_metrics.get('avg_rouge1', 0):.3f}")
            print(f"Average ROUGE-L: {accuracy_metrics.get('avg_rougeL', 0):.3f}")
            print(f"Average BLEU: {accuracy_metrics.get('avg_bleu', 0):.3f}")
        
        if memory_metrics:
            print(f"Model size: {memory_metrics.get('model_size_mb', 0):.1f}MB")
            print(f"Memory utilization: {memory_metrics.get('memory_utilization_percent', 0):.1f}%")
        
        print(f"Total evaluation time: {total_time:.2f}s")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
