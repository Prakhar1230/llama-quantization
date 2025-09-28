import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from datasets import load_dataset
import time
import json
import logging
from typing import List, Dict, Any
import psutil
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive evaluation of quantized models."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def load_model_for_evaluation(self, model_path: str):
        """Load quantized model for evaluation with proper torch.load settings."""
        try:
            logger.info(f"Loading quantized model from {model_path}")
            
            # Check what type of quantized model we have
            quantized_model_file = os.path.join(model_path, "quantized_model.pt")
            
            # Load tokenizer (this should always work)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if os.path.exists(quantized_model_file):
                # Load full quantized model with safe_globals
                logger.info("Loading full quantized model...")
                
                # Add safe globals for LlamaForCausalLM
                torch.serialization.add_safe_globals([LlamaForCausalLM])
                
                # Alternative approach: use weights_only=False (since we trust our own model)
                try:
                    # Try with safe_globals first (more secure)
                    with torch.serialization.safe_globals([LlamaForCausalLM]):
                        model = torch.load(quantized_model_file, map_location='cpu', weights_only=True)
                    logger.info("Model loaded with safe_globals")
                except Exception as e:
                    logger.warning(f"Safe loading failed: {e}")
                    logger.info("Falling back to weights_only=False (trusted source)")
                    # Fallback: since we created this model ourselves, it's safe
                    model = torch.load(quantized_model_file, map_location='cpu', weights_only=False)
                    logger.info("Model loaded with weights_only=False")
                
                logger.info("Quantized model loaded successfully")
                
            else:
                # Fallback: try to load as regular model
                logger.info("No quantized model files found, trying standard loading...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_path)
                    logger.info("Standard model loaded successfully")
                except Exception as e:
                    raise FileNotFoundError(f"No compatible model found in {model_path}. Error: {str(e)}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise
    
    def prepare_evaluation_data(self, dataset_name: str = "wikitext", num_samples: int = 100) -> List[Dict]:
        """Prepare evaluation dataset."""
        logger.info(f"Preparing evaluation data from {dataset_name}")
        
        try:
            if dataset_name == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            else:
                dataset = load_dataset("c4", "en", split="validation", streaming=True)
            
            evaluation_data = []
            
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                
                text = sample.get("text", "")
                
                if len(text.strip()) > 100:  # Filter short texts
                    # Create prompt-completion pairs
                    sentences = text.split('. ')
                    if len(sentences) >= 2:
                        mid_point = len(sentences) // 2
                        prompt = '. '.join(sentences[:mid_point]) + '.'
                        completion = '. '.join(sentences[mid_point:])
                        
                        evaluation_data.append({
                            "prompt": prompt[:200],  # Limit prompt length
                            "reference": completion[:200],  # Limit reference length
                            "full_text": text
                        })
            
            logger.info(f"Prepared {len(evaluation_data)} evaluation samples")
            return evaluation_data
            
        except Exception as e:
            logger.error(f"Error preparing evaluation data: {str(e)}")
            # Fallback to dummy data
            return [
                {
                    "prompt": "What is artificial intelligence?",
                    "reference": "Artificial intelligence is a field of computer science.",
                    "full_text": "What is artificial intelligence? Artificial intelligence is a field of computer science."
                },
                {
                    "prompt": "Explain machine learning.",
                    "reference": "Machine learning is a subset of AI that enables computers to learn.",
                    "full_text": "Explain machine learning. Machine learning is a subset of AI that enables computers to learn."
                }
            ]
    
    def evaluate_inference_speed(self, model, tokenizer, test_prompts: List[str], max_length: int = 64) -> Dict[str, float]:
        """Evaluate inference speed metrics."""
        logger.info("Evaluating inference speed...")
        
        inference_times = []
        tokens_per_second_list = []
        successful_runs = 0
        
        # Reduce max_length for faster evaluation on CPU
        max_length = min(max_length, 32)
        
        for i, prompt in enumerate(test_prompts[:5]):  # Further limit to 5 prompts for CPU
            try:
                logger.info(f"Processing prompt {i+1}/5: {prompt[:50]}...")
                
                start_time = time.time()
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_length = inputs['input_ids'].shape[1]
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        max_length=min(max_length + input_length, 96),
                        do_sample=False,  # Deterministic for consistency
                        pad_token_id=tokenizer.eos_token_id,
                        num_beams=1,  # Faster generation
                        early_stopping=True
                    )
                
                inference_time = time.time() - start_time
                output_length = outputs.shape[1] - input_length
                
                if inference_time > 0 and output_length > 0:
                    tokens_per_second = output_length / inference_time
                    tokens_per_second_list.append(tokens_per_second)
                
                inference_times.append(inference_time)
                successful_runs += 1
                
                logger.info(f"  Time: {inference_time:.2f}s, Tokens: {output_length}, Speed: {output_length/inference_time:.1f} tok/s")
                
            except Exception as e:
                logger.warning(f"Error in speed evaluation for prompt {i}: {str(e)}")
                continue
        
        if not inference_times:
            logger.error("No successful inference runs!")
            return {
                "avg_inference_time": 0,
                "median_inference_time": 0,
                "avg_tokens_per_second": 0,
                "total_prompts_evaluated": 0
            }
        
        return {
            "avg_inference_time": np.mean(inference_times),
            "median_inference_time": np.median(inference_times),
            "std_inference_time": np.std(inference_times),
            "min_inference_time": np.min(inference_times),
            "max_inference_time": np.max(inference_times),
            "avg_tokens_per_second": np.mean(tokens_per_second_list) if tokens_per_second_list else 0,
            "total_prompts_evaluated": successful_runs
        }
    
    def evaluate_accuracy(self, model, tokenizer, evaluation_data: List[Dict], max_length: int = 32) -> Dict[str, float]:
        """Evaluate model accuracy using ROUGE and BLEU scores."""
        logger.info("Evaluating model accuracy...")
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        bleu_scores = []
        successful_evaluations = 0
        
        for i, data in enumerate(evaluation_data[:5]):  # Limit for CPU
            try:
                logger.info(f"Evaluating accuracy for sample {i+1}/5")
                
                prompt = data['prompt']
                reference = data['reference']
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        max_length=min(max_length + inputs['input_ids'].shape[1], 96),
                        do_sample=False,
                        temperature=0.1,  # Lower temperature for consistency
                        pad_token_id=tokenizer.eos_token_id,
                        num_beams=1,
                        early_stopping=True
                    )
                
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated = generated[len(prompt):].strip()
                
                logger.info(f"  Prompt: {prompt[:50]}...")
                logger.info(f"  Generated: {generated[:50]}...")
                logger.info(f"  Reference: {reference[:50]}...")
                
                if len(generated) > 0:
                    # Calculate ROUGE scores
                    rouge_score = self.rouge_scorer.score(reference, generated)
                    for metric in rouge_scores:
                        rouge_scores[metric].append(rouge_score[metric].fmeasure)
                    
                    # Calculate BLEU score
                    reference_tokens = reference.split()
                    generated_tokens = generated.split()
                    
                    if len(generated_tokens) > 0:
                        bleu_score = sentence_bleu(
                            [reference_tokens], 
                            generated_tokens, 
                            smoothing_function=self.smoothing_function.method1
                        )
                        bleu_scores.append(bleu_score)
                
                successful_evaluations += 1
                
            except Exception as e:
                logger.warning(f"Error in accuracy evaluation for sample {i}: {str(e)}")
                continue
        
        if successful_evaluations == 0:
            logger.error("No successful accuracy evaluations!")
            return {
                "avg_rouge1": 0,
                "avg_rouge2": 0,
                "avg_rougeL": 0,
                "avg_bleu": 0,
                "samples_evaluated": 0
            }
        
        return {
            "avg_rouge1": np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0,
            "avg_rouge2": np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0,
            "avg_rougeL": np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0,
            "avg_bleu": np.mean(bleu_scores) if bleu_scores else 0,
            "samples_evaluated": successful_evaluations
        }
    
    def evaluate_memory_usage(self, model) -> Dict[str, float]:
        """Evaluate memory consumption."""
        logger.info("Evaluating memory usage...")
        
        # Model memory
        try:
            if hasattr(model, 'parameters'):
                model_size = sum(p.numel() * p.element_size() for p in model.parameters() if hasattr(p, 'numel')) / (1024 * 1024)
            else:
                # For quantized models, estimate from file size
                model_size = 0
        except:
            model_size = 0
        
        # System memory
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "model_size_mb": model_size,
            "process_memory_mb": memory_info.rss / (1024 * 1024),
            "available_memory_mb": psutil.virtual_memory().available / (1024 * 1024),
            "memory_utilization_percent": psutil.virtual_memory().percent
        }
    
    def comprehensive_evaluation(self, model_path: str, output_path: str = "evaluation_results.json") -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        logger.info("Starting comprehensive evaluation...")
        
        try:
            # Load model
            model, tokenizer = self.load_model_for_evaluation(model_path)
            
            # Prepare evaluation data
            evaluation_data = self.prepare_evaluation_data(num_samples=10)
            test_prompts = [data['prompt'] for data in evaluation_data]
            
            # Run evaluations with reduced scope for CPU
            logger.info("Running speed evaluation...")
            speed_metrics = self.evaluate_inference_speed(model, tokenizer, test_prompts, max_length=32)
            
            logger.info("Running accuracy evaluation...")
            accuracy_metrics = self.evaluate_accuracy(model, tokenizer, evaluation_data, max_length=32)
            
            logger.info("Running memory evaluation...")
            memory_metrics = self.evaluate_memory_usage(model)
            
            # Compile results
            results = {
                "model_path": model_path,
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_type": "quantized" if "quantized" in model_path else "standard",
                "speed_metrics": speed_metrics,
                "accuracy_metrics": accuracy_metrics,
                "memory_metrics": memory_metrics,
                "evaluation_config": {
                    "max_generation_length": 32,
                    "num_speed_samples": speed_metrics.get("total_prompts_evaluated", 0),
                    "num_accuracy_samples": accuracy_metrics.get("samples_evaluated", 0),
                    "pytorch_version": torch.__version__
                }
            }
            
            # Save results
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Evaluation completed. Results saved to {output_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            print(f"Speed Metrics:")
            print(f"  Average inference time: {speed_metrics.get('avg_inference_time', 0):.3f}s")
            print(f"  Tokens per second: {speed_metrics.get('avg_tokens_per_second', 0):.2f}")
            print(f"  Prompts evaluated: {speed_metrics.get('total_prompts_evaluated', 0)}")
            
            print(f"\nAccuracy Metrics:")
            print(f"  ROUGE-1: {accuracy_metrics.get('avg_rouge1', 0):.3f}")
            print(f"  ROUGE-L: {accuracy_metrics.get('avg_rougeL', 0):.3f}")
            print(f"  BLEU: {accuracy_metrics.get('avg_bleu', 0):.3f}")
            print(f"  Samples evaluated: {accuracy_metrics.get('samples_evaluated', 0)}")
            
            print(f"\nMemory Metrics:")
            print(f"  Model size: {memory_metrics.get('model_size_mb', 0):.1f} MB")
            print(f"  Process memory: {memory_metrics.get('process_memory_mb', 0):.1f} MB")
            print(f"  Memory utilization: {memory_metrics.get('memory_utilization_percent', 0):.1f}%")
            print("="*60)
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {str(e)}")
            raise
