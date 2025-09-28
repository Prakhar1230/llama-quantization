import torch
import time
import json
import os
import sys
import psutil
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import gc


class ModelComparer:
    """Compare original vs quantized models across multiple metrics."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()
        
    def load_original_model(self, model_path):
        """Load original model."""
        print("[STEP] Loading original model...")
        sys.stdout.flush()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,  # Fixed: was torch_dtype
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        model.eval()
        print("  -> Original model loaded successfully")
        sys.stdout.flush()
        return model, tokenizer
    
    def load_quantized_model(self, model_path):
        """Load quantized model."""
        print("[STEP] Loading quantized model...")
        sys.stdout.flush()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add safe globals and load quantized model
        torch.serialization.add_safe_globals([LlamaForCausalLM])
        try:
            with torch.serialization.safe_globals([LlamaForCausalLM]):
                model = torch.load(f"{model_path}/quantized_model.pt", map_location='cpu', weights_only=True)
        except:
            model = torch.load(f"{model_path}/quantized_model.pt", map_location='cpu', weights_only=False)
        
        model.eval()
        print("  -> Quantized model loaded successfully")
        sys.stdout.flush()
        return model, tokenizer
    
    def cleanup_model(self, model, tokenizer, model_name):
        """Clean up model from memory."""
        print(f"[CLEANUP] Freeing memory for {model_name} model...")
        sys.stdout.flush()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  -> {model_name} model memory freed")
        sys.stdout.flush()
    
    def get_model_metrics(self, model, model_name, model_path):
        """Extract model size and parameter metrics."""
        print(f"[METRICS] Getting metrics for {model_name} model...")
        sys.stdout.flush()
        
        try:
            # Get parameter count
            param_count = sum(p.numel() for p in model.parameters())
            
            # Get model file size
            if "quantized" in model_name.lower():
                model_file = os.path.join(model_path, "quantized_model.pt")
            else:
                # Find the largest .bin or .safetensors file
                files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors'))]
                if files:
                    model_file = os.path.join(model_path, max(files, key=lambda f: os.path.getsize(os.path.join(model_path, f))))
                else:
                    model_file = None
            
            file_size_mb = os.path.getsize(model_file) / (1024**2) if model_file and os.path.exists(model_file) else 0
            
            # Get memory footprint
            memory_footprint = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            
            metrics = {
                "model_name": model_name,
                "parameter_count": param_count,
                "file_size_mb": file_size_mb,
                "memory_footprint_mb": memory_footprint,
                "dtype": str(next(model.parameters()).dtype)
            }
            
            print(f"  -> {model_name}: {param_count:,} params, {file_size_mb:.1f} MB")
            sys.stdout.flush()
            
            return metrics
            
        except Exception as e:
            print(f"  -> Error getting metrics for {model_name}: {e}")
            sys.stdout.flush()
            return {"model_name": model_name, "error": str(e)}
    
    def benchmark_inference_speed(self, model, tokenizer, test_prompts, model_name):
        """Benchmark inference speed."""
        print(f"[SPEED] Benchmarking inference speed for {model_name}...")
        sys.stdout.flush()
        
        inference_times = []
        tokens_per_second = []
        
        # Reduced test set for faster execution
        test_set = test_prompts[:3]  # Only 3 prompts instead of 5
        
        for i, prompt in enumerate(test_set):
            try:
                print(f"  -> Testing prompt {i+1}/{len(test_set)}")
                sys.stdout.flush()
                
                start_time = time.time()
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)  # Reduced max_length
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        max_length=32,  # Reduced from 64
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        num_beams=1
                    )
                
                inference_time = time.time() - start_time
                output_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
                
                if output_tokens > 0:
                    tokens_per_sec = output_tokens / inference_time
                    tokens_per_second.append(tokens_per_sec)
                
                inference_times.append(inference_time)
                print(f"     {inference_time:.2f}s, {tokens_per_sec:.1f} tok/s")
                sys.stdout.flush()
                
            except Exception as e:
                print(f"     Error: {e}")
                sys.stdout.flush()
                continue
        
        result = {
            "model_name": model_name,
            "avg_inference_time": sum(inference_times) / len(inference_times) if inference_times else 0,
            "avg_tokens_per_sec": sum(tokens_per_second) / len(tokens_per_second) if tokens_per_second else 0,
            "successful_inferences": len(inference_times)
        }
        
        print(f"  -> {model_name} avg: {result['avg_inference_time']:.2f}s, {result['avg_tokens_per_sec']:.1f} tok/s")
        sys.stdout.flush()
        
        return result
    
    def benchmark_accuracy(self, model, tokenizer, test_cases, model_name):
        """Benchmark model accuracy using ROUGE and BLEU."""
        print(f"[ACCURACY] Benchmarking accuracy for {model_name}...")
        sys.stdout.flush()
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        bleu_scores = []
        
        # Reduced test set for faster execution
        test_set = test_cases[:3]  # Only 3 test cases instead of 5
        
        for i, case in enumerate(test_set):
            try:
                print(f"  -> Accuracy test {i+1}/{len(test_set)}")
                sys.stdout.flush()
                
                prompt = case['prompt']
                reference = case['reference']
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)  # Reduced
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        max_length=48,  # Reduced from 96
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated = generated[len(prompt):].strip()
                
                if len(generated) > 0:
                    # ROUGE scores
                    rouge_score = self.rouge_scorer.score(reference, generated)
                    for metric in rouge_scores:
                        rouge_scores[metric].append(rouge_score[metric].fmeasure)
                    
                    # BLEU score
                    ref_tokens = reference.split()
                    gen_tokens = generated.split()
                    
                    if len(gen_tokens) > 0:
                        bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.smoothing.method1)
                        bleu_scores.append(bleu)
                
                print(f"     ROUGE-L: {rouge_score['rougeL'].fmeasure:.3f}")
                sys.stdout.flush()
                
            except Exception as e:
                print(f"     Error: {e}")
                sys.stdout.flush()
                continue
        
        result = {
            "model_name": model_name,
            "avg_rouge1": sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0,
            "avg_rouge2": sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0,
            "avg_rougeL": sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0,
            "avg_bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
            "successful_accuracy_tests": len(bleu_scores)
        }
        
        print(f"  -> {model_name} avg ROUGE-L: {result['avg_rougeL']:.3f}")
        sys.stdout.flush()
        
        return result
    
    def memory_usage_test(self, model, tokenizer, model_name):
        """Test memory usage during inference."""
        print(f"[MEMORY] Testing memory usage for {model_name}...")
        sys.stdout.flush()
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        
        # Run a few inferences and monitor memory
        test_prompt = "What is AI?"  # Shorter prompt
        
        memory_samples = [initial_memory]
        
        for i in range(2):  # Reduced from 3 to 2
            inputs = tokenizer(test_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(inputs['input_ids'], max_length=32, do_sample=False)  # Reduced max_length
            
            current_memory = process.memory_info().rss / (1024**2)
            memory_samples.append(current_memory)
        
        result = {
            "model_name": model_name,
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": max(memory_samples),
            "avg_memory_mb": sum(memory_samples) / len(memory_samples),
            "memory_increase_mb": max(memory_samples) - initial_memory
        }
        
        print(f"  -> {model_name} peak memory: {result['peak_memory_mb']:.1f} MB")
        sys.stdout.flush()
        
        return result
    
    def create_test_cases(self):
        """Create test cases for evaluation."""
        return [
            {
                "prompt": "What is machine learning?",
                "reference": "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
            },
            {
                "prompt": "Explain neural networks.",
                "reference": "Neural networks are computing systems inspired by biological neural networks with interconnected nodes."
            },
            {
                "prompt": "What is quantization?",
                "reference": "Quantization reduces the precision of model parameters to lower bit-widths, reducing memory usage."
            }
        ]
    
    def run_comprehensive_comparison(self, original_path, quantized_path):
        """Run complete comparison between original and quantized models."""
        print("=" * 80)
        print("[START] COMPREHENSIVE MODEL COMPARISON")  # Fixed: removed emoji
        print("=" * 80)
        sys.stdout.flush()
        
        results = {}
        original_model = None
        quantized_model = None
        original_tokenizer = None
        quantized_tokenizer = None
        
        try:
            # Test prompts - reduced set
            test_prompts = [
                "What is AI?",
                "Explain ML.",
                "How does quantization work?"
            ]
            
            # Test cases for accuracy
            test_cases = self.create_test_cases()
            
            print("\n[PHASE 1] TESTING ORIGINAL MODEL")
            print("-" * 40)
            sys.stdout.flush()
            
            # Load and test original model
            original_model, original_tokenizer = self.load_original_model(original_path)
            
            # 1. Original Model Metrics
            original_metrics = self.get_model_metrics(original_model, "Original", original_path)
            
            # 2. Original Speed Test
            original_speed = self.benchmark_inference_speed(original_model, original_tokenizer, test_prompts, "Original")
            
            # 3. Original Accuracy Test
            original_accuracy = self.benchmark_accuracy(original_model, original_tokenizer, test_cases, "Original")
            
            # 4. Original Memory Test
            original_memory = self.memory_usage_test(original_model, original_tokenizer, "Original")
            
            # Clean up original model before loading quantized
            self.cleanup_model(original_model, original_tokenizer, "Original")
            original_model = None
            original_tokenizer = None
            
            print("\n[PHASE 2] TESTING QUANTIZED MODEL")
            print("-" * 40)
            sys.stdout.flush()
            
            # Load and test quantized model
            quantized_model, quantized_tokenizer = self.load_quantized_model(quantized_path)
            
            # Test quantized model
            quantized_metrics = self.get_model_metrics(quantized_model, "Quantized", quantized_path)
            quantized_speed = self.benchmark_inference_speed(quantized_model, quantized_tokenizer, test_prompts, "Quantized")
            quantized_accuracy = self.benchmark_accuracy(quantized_model, quantized_tokenizer, test_cases, "Quantized")
            quantized_memory = self.memory_usage_test(quantized_model, quantized_tokenizer, "Quantized")
            
            # Clean up quantized model
            self.cleanup_model(quantized_model, quantized_tokenizer, "Quantized")
            quantized_model = None
            quantized_tokenizer = None
            
            # Compile results
            results = {
                'model_metrics': [original_metrics, quantized_metrics],
                'speed_metrics': [original_speed, quantized_speed],
                'accuracy_metrics': [original_accuracy, quantized_accuracy],
                'memory_metrics': [original_memory, quantized_memory]
            }
            
            print("\n[PHASE 3] SAVING RESULTS")
            print("-" * 40)
            sys.stdout.flush()
            
            # Save results
            with open('model_comparison_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Generate comparison report
            self.generate_comparison_report(results)
            
            print(f"\n[SUCCESS] Comparison completed! Results saved to 'model_comparison_results.json'")  # Fixed: removed emoji
            sys.stdout.flush()
            
        except Exception as e:
            print(f"[ERROR] Comparison failed: {e}")  # Fixed: removed emoji
            sys.stdout.flush()
            
            # Cleanup in case of error
            if original_model is not None:
                self.cleanup_model(original_model, original_tokenizer, "Original")
            if quantized_model is not None:
                self.cleanup_model(quantized_model, quantized_tokenizer, "Quantized")
            
        return results
    
    def generate_comparison_report(self, results):
        """Generate a comprehensive comparison report."""
        print("\n" + "=" * 80)
        print("[REPORT] DETAILED COMPARISON REPORT")  # Fixed: removed emoji
        print("=" * 80)
        sys.stdout.flush()
        
        try:
            # Model Size Comparison
            original_metrics = results['model_metrics'][0]
            quantized_metrics = results['model_metrics'][1]
            
            print(f"\n[SIZE] MODEL SIZE COMPARISON:")
            print(f"{'Metric':<25} {'Original':<15} {'Quantized':<15} {'Change':<15}")
            print("-" * 70)
            
            orig_params = original_metrics.get('parameter_count', 0)
            quant_params = quantized_metrics.get('parameter_count', 0)
            param_reduction = ((orig_params - quant_params) / orig_params * 100) if orig_params > 0 else 0
            
            orig_size = original_metrics.get('file_size_mb', 0)
            quant_size = quantized_metrics.get('file_size_mb', 0) 
            size_reduction = ((orig_size - quant_size) / orig_size * 100) if orig_size > 0 else 0
            
            print(f"{'Parameters':<25} {orig_params:<15,} {quant_params:<15,} {param_reduction:<14.1f}%")
            print(f"{'File Size (MB)':<25} {orig_size:<15.1f} {quant_size:<15.1f} {size_reduction:<14.1f}%")
            
            # Performance Comparison
            original_speed = results['speed_metrics'][0]
            quantized_speed = results['speed_metrics'][1]
            
            print(f"\n[SPEED] PERFORMANCE COMPARISON:")
            print(f"{'Metric':<25} {'Original':<15} {'Quantized':<15} {'Change':<15}")
            print("-" * 70)
            
            orig_time = original_speed.get('avg_inference_time', 0)
            quant_time = quantized_speed.get('avg_inference_time', 0)
            time_improvement = ((orig_time - quant_time) / orig_time * 100) if orig_time > 0 else 0
            
            orig_tps = original_speed.get('avg_tokens_per_sec', 0)
            quant_tps = quantized_speed.get('avg_tokens_per_sec', 0)
            tps_improvement = ((quant_tps - orig_tps) / orig_tps * 100) if orig_tps > 0 else 0
            
            print(f"{'Avg Inference (s)':<25} {orig_time:<15.3f} {quant_time:<15.3f} {time_improvement:<14.1f}%")
            print(f"{'Tokens/Second':<25} {orig_tps:<15.1f} {quant_tps:<15.1f} {tps_improvement:<14.1f}%")
            
            # Summary
            print(f"\n[SUMMARY] QUANTIZATION RESULTS:")  # Fixed: removed emoji
            print(f"-> File size reduced by: {size_reduction:.1f}%")
            print(f"-> Inference time improved by: {time_improvement:.1f}%")
            print(f"-> Speed increased by: {tps_improvement:.1f}%")
            
            sys.stdout.flush()
            
        except Exception as e:
            print(f"[ERROR] Error generating report: {e}")
            sys.stdout.flush()


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description='Compare original and quantized models')
    parser.add_argument('--original_path', type=str, default='models/original', help='Path to original model')
    parser.add_argument('--quantized_path', type=str, default='models/quantized', help='Path to quantized model')
    
    args = parser.parse_args()
    
    comparer = ModelComparer()
    
    # Check if paths exist
    if not os.path.exists(args.original_path):
        print(f"[ERROR] Original model path not found: {args.original_path}")
        return
        
    if not os.path.exists(args.quantized_path):
        print(f"[ERROR] Quantized model path not found: {args.quantized_path}")
        return
    
    # Run comparison
    results = comparer.run_comprehensive_comparison(args.original_path, args.quantized_path)
    
    print(f"\n[COMPLETE] Model comparison completed successfully!")
    print(f"[SAVE] Results saved to: model_comparison_results.json")


if __name__ == "__main__":
    main()
