#!/usr/bin/env python3

import argparse
import os
import sys
import logging
from pathlib import Path
import time
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_loader import ModelLoader
from utils import BenchmarkUtils, MemoryMonitor, FileUtils
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelBenchmark:
    """Comprehensive benchmarking for quantized models."""
    
    def __init__(self):
        self.memory_monitor = MemoryMonitor()
        self.benchmark_utils = BenchmarkUtils()
        
    def run_latency_benchmark(self, model, tokenizer, test_prompts, 
                            max_length=128, num_runs=5):
        """Benchmark inference latency."""
        logger.info("Running latency benchmark...")
        
        latency_results = self.benchmark_utils.measure_inference_time(
            model, tokenizer, test_prompts, max_length, num_runs
        )
        
        return latency_results
    
    def run_throughput_benchmark(self, model, tokenizer, test_prompts, 
                                max_length=128, batch_sizes=[1]):
        """Benchmark throughput with different batch sizes."""
        logger.info("Running throughput benchmark...")
        
        throughput_results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Take subset of prompts for batch
            batch_prompts = test_prompts[:batch_size]
            
            start_time = time.time()
            total_tokens = 0
            
            try:
                for prompt in batch_prompts:
                    inputs = tokenizer(prompt, return_tensors="pt", 
                                     truncation=True, max_length=256)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs["input_ids"],
                            max_length=max_length,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # Count generated tokens
                    input_tokens = inputs["input_ids"].shape[1]
                    output_tokens = outputs.shape[1] - input_tokens
                    total_tokens += output_tokens
                
                total_time = time.time() - start_time
                
                throughput_results[f"batch_{batch_size}"] = {
                    "total_time": total_time,
                    "total_tokens": total_tokens,
                    "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
                    "samples_per_second": len(batch_prompts) / total_time if total_time > 0 else 0
                }
                
            except Exception as e:
                logger.error(f"Error in throughput benchmark for batch size {batch_size}: {str(e)}")
                continue
        
        return throughput_results
    
    def run_memory_benchmark(self, model, tokenizer, test_prompts, max_length=128):
        """Benchmark memory usage during inference."""
        logger.info("Running memory benchmark...")
        
        initial_memory = self.memory_monitor.get_current_memory()
        memory_samples = [initial_memory]
        
        for i, prompt in enumerate(test_prompts[:5]):  # Limited for CPU
            try:
                inputs = tokenizer(prompt, return_tensors="pt", 
                                 truncation=True, max_length=256)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                current_memory = self.memory_monitor.get_current_memory()
                memory_samples.append(current_memory)
                
            except Exception as e:
                logger.error(f"Error in memory benchmark for prompt {i}: {str(e)}")
                continue
        
        # Calculate memory statistics
        rss_values = [m["rss_mb"] for m in memory_samples]
        
        memory_stats = {
            "initial_memory_mb": initial_memory["rss_mb"],
            "peak_memory_mb": max(rss_values),
            "avg_memory_mb": sum(rss_values) / len(rss_values),
            "memory_increase_mb": max(rss_values) - initial_memory["rss_mb"],
            "memory_samples": len(memory_samples),
            "available_memory_mb": memory_samples[-1]["available_mb"]
        }
        
        return memory_stats
    
    def run_accuracy_benchmark(self, model, tokenizer, test_data, max_length=128):
        """Run accuracy benchmark with predefined test cases."""
        logger.info("Running accuracy benchmark...")
        
        correct_predictions = 0
        total_predictions = 0
        response_lengths = []
        
        for test_case in test_data:
            try:
                prompt = test_case.get("prompt", "")
                expected_keywords = test_case.get("expected_keywords", [])
                
                inputs = tokenizer(prompt, return_tensors="pt", 
                                 truncation=True, max_length=256)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        temperature=0.1,  # Lower temperature for consistency
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated[len(prompt):].strip().lower()
                
                # Check if expected keywords are in response
                keyword_matches = sum(1 for keyword in expected_keywords 
                                    if keyword.lower() in response)
                
                if keyword_matches > 0:
                    correct_predictions += 1
                
                total_predictions += 1
                response_lengths.append(len(response.split()))
                
            except Exception as e:
                logger.error(f"Error in accuracy benchmark: {str(e)}")
                continue
        
        accuracy_stats = {
            "accuracy": correct_predictions / total_predictions if total_predictions > 0 else 0,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "avg_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            "max_response_length": max(response_lengths) if response_lengths else 0,
            "min_response_length": min(response_lengths) if response_lengths else 0
        }
        
        return accuracy_stats

def create_test_prompts():
    """Create standardized test prompts for benchmarking."""
    return [
        "Explain the concept of artificial intelligence in simple terms.",
        "What are the main benefits of renewable energy?",
        "Describe the process of photosynthesis.",
        "How does machine learning work?",
        "What is the importance of data privacy?",
        "Explain quantum computing to a beginner.",
        "What are the effects of climate change?",
        "How do neural networks function?",
        "What is the role of cybersecurity?",
        "Describe the benefits of cloud computing."
    ]

def create_accuracy_test_data():
    """Create test cases for accuracy evaluation."""
    return [
        {
            "prompt": "What is machine learning?",
            "expected_keywords": ["algorithm", "data", "pattern", "prediction", "model"]
        },
        {
            "prompt": "Explain photosynthesis.",
            "expected_keywords": ["plants", "sunlight", "carbon dioxide", "oxygen", "energy"]
        },
        {
            "prompt": "What is artificial intelligence?",
            "expected_keywords": ["computer", "human", "intelligence", "automation", "algorithm"]
        },
        {
            "prompt": "How does the internet work?",
            "expected_keywords": ["network", "data", "protocol", "server", "connection"]
        },
        {
            "prompt": "What is renewable energy?",
            "expected_keywords": ["solar", "wind", "sustainable", "clean", "environment"]
        }
    ]

def main():
    parser = argparse.ArgumentParser(description="Comprehensive model benchmarking")
    parser.add_argument("--model_path", required=True, help="Path to model to benchmark")
    parser.add_argument("--output_path", default="results/benchmark.json",
                       help="Path to save benchmark results")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum generation length")
    parser.add_argument("--num_runs", type=int, default=3,
                       help="Number of runs for latency benchmark")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1],
                       help="Batch sizes to test for throughput")
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting comprehensive model benchmark...")
        start_time = time.time()
        
        # Initialize benchmark
        benchmark = ModelBenchmark()
        
        # Load model
        model_loader = ModelLoader()
        
        with benchmark.memory_monitor.track_memory("model_loading"):
            model, tokenizer = model_loader.load_model_and_tokenizer(args.model_path)
            model_info = model_loader.get_model_info(model)
        
        # Create test data
        test_prompts = create_test_prompts()
        accuracy_test_data = create_accuracy_test_data()
        
        # Run benchmarks
        benchmark_results = {
            "model_path": args.model_path,
            "model_info": model_info,
            "benchmark_config": {
                "max_length": args.max_length,
                "num_runs": args.num_runs,
                "batch_sizes": args.batch_sizes,
                "num_test_prompts": len(test_prompts)
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Latency benchmark
        with benchmark.memory_monitor.track_memory("latency_benchmark"):
            latency_results = benchmark.run_latency_benchmark(
                model, tokenizer, test_prompts, args.max_length, args.num_runs
            )
            benchmark_results["latency"] = latency_results
        
        # Throughput benchmark
        with benchmark.memory_monitor.track_memory("throughput_benchmark"):
            throughput_results = benchmark.run_throughput_benchmark(
                model, tokenizer, test_prompts, args.max_length, args.batch_sizes
            )
            benchmark_results["throughput"] = throughput_results
        
        # Memory benchmark
        memory_results = benchmark.run_memory_benchmark(
            model, tokenizer, test_prompts, args.max_length
        )
        benchmark_results["memory"] = memory_results
        
        # Accuracy benchmark
        with benchmark.memory_monitor.track_memory("accuracy_benchmark"):
            accuracy_results = benchmark.run_accuracy_benchmark(
                model, tokenizer, accuracy_test_data, args.max_length
            )
            benchmark_results["accuracy"] = accuracy_results
        
        # Add total benchmark time
        total_time = time.time() - start_time
        benchmark_results["total_benchmark_time"] = total_time
        
        # Save results
        FileUtils.ensure_dir_exists(Path(args.output_path).parent)
        FileUtils.save_json(benchmark_results, args.output_path)
        
        # Print summary
        logger.info("Benchmark completed successfully!")
        logger.info(f"Results saved to: {args.output_path}")
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"Model: {args.model_path}")
        print(f"Parameters: {model_info.get('num_parameters', 0):,}")
        print(f"Model size: {model_info.get('model_size_mb', 0):.1f}MB")
        
        if latency_results:
            print(f"\nLatency:")
            print(f"  Average: {latency_results.get('mean_time', 0):.3f}s")
            print(f"  Median: {latency_results.get('median_time', 0):.3f}s")
            print(f"  Tokens/sec: {latency_results.get('mean_tokens_generated', 0):.1f}")
        
        if throughput_results:
            print(f"\nThroughput:")
            for batch, results in throughput_results.items():
                print(f"  {batch}: {results.get('tokens_per_second', 0):.1f} tokens/sec")
        
        if memory_results:
            print(f"\nMemory:")
            print(f"  Peak: {memory_results.get('peak_memory_mb', 0):.1f}MB")
            print(f"  Average: {memory_results.get('avg_memory_mb', 0):.1f}MB")
            print(f"  Increase: {memory_results.get('memory_increase_mb', 0):.1f}MB")
        
        if accuracy_results:
            print(f"\nAccuracy:")
            print(f"  Score: {accuracy_results.get('accuracy', 0):.1%}")
            print(f"  Correct: {accuracy_results.get('correct_predictions', 0)}/{accuracy_results.get('total_predictions', 0)}")
        
        print(f"\nTotal benchmark time: {total_time:.2f}s")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error during benchmark: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
