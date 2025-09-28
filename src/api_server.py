from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import logging
import time
import os
import uvicorn
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
model_info = {}

class InferenceRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True

class InferenceResponse(BaseModel):
    generated_text: str
    inference_time: float
    tokens_per_second: float
    input_tokens: int
    output_tokens: int

def load_quantized_model(model_path: str):
    """Load quantized model with proper error handling."""
    global model, tokenizer, model_info
    
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load tokenizer first (this should always work)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("✓ Tokenizer loaded successfully")
        
        # Check for quantized model file
        quantized_model_file = os.path.join(model_path, "quantized_model.pt")
        
        if os.path.exists(quantized_model_file):
            logger.info("Found quantized model file, loading...")
            
            # Add safe globals for secure loading
            torch.serialization.add_safe_globals([LlamaForCausalLM])
            
            try:
                # Try secure loading first
                with torch.serialization.safe_globals([LlamaForCausalLM]):
                    model = torch.load(quantized_model_file, map_location='cpu', weights_only=True)
                logger.info("✓ Quantized model loaded securely with safe_globals")
            except Exception as e:
                logger.warning(f"Safe loading failed: {e}")
                logger.info("Falling back to trusted loading...")
                # Fallback to trusted loading (our own model is safe)
                model = torch.load(quantized_model_file, map_location='cpu', weights_only=False)
                logger.info("✓ Quantized model loaded with weights_only=False")
            
            # Set model to evaluation mode
            model.eval()
            
            # Get model info
            try:
                model_size_mb = os.path.getsize(quantized_model_file) / (1024 * 1024)
                param_count = sum(p.numel() for p in model.parameters() if hasattr(p, 'numel'))
                
                model_info = {
                    "model_type": "quantized",
                    "model_file": quantized_model_file,
                    "model_size_mb": model_size_mb,
                    "num_parameters": param_count,
                    "device": str(next(model.parameters()).device),
                    "dtype": str(next(model.parameters()).dtype)
                }
            except Exception as e:
                logger.warning(f"Could not extract model info: {e}")
                model_info = {
                    "model_type": "quantized",
                    "model_file": quantized_model_file,
                    "error": str(e)
                }
            
        else:
            # Fallback: try to load as standard model
            logger.info("No quantized model file found, trying standard loading...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )
                model.eval()
                logger.info("✓ Standard model loaded successfully")
                
                model_info = {
                    "model_type": "standard",
                    "num_parameters": model.num_parameters(),
                    "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
                    "device": str(next(model.parameters()).device),
                    "dtype": str(next(model.parameters()).dtype)
                }
                
            except Exception as e:
                raise RuntimeError(f"Could not load model from {model_path}. Error: {str(e)}")
        
        logger.info("Model loading completed successfully!")
        logger.info(f"Model info: {model_info}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
def get_host():
    """Auto-detect if running in Docker and set appropriate host."""
    if os.path.exists('/.dockerenv') or 'docker' in str(os.environ.get('container', '')):
        return "0.0.0.0"  # Docker - bind to all interfaces
    else:
        return "127.0.0.1"  # Local - bind to localhost only

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    model_path = os.getenv("MODEL_PATH", "models/quantized")
    
    try:
        load_quantized_model(model_path)
        logger.info("Application startup completed successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
    
    # Cleanup on shutdown (optional)
    logger.info("Application shutting down...")

app = FastAPI(
    title="LLM Quantization API",
    description="CPU-optimized quantized LLM inference API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "model_type": model_info.get("model_type", "unknown")
    }

@app.post("/inference", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    """Generate text using the quantized model."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, max_length=256)
        input_length = inputs['input_ids'].shape[1]
        
        # Generate text with optimized parameters for CPU
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=min(request.max_length + input_length, 256),  # Limit total length
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,  # Faster for CPU
                early_stopping=True,
                no_repeat_ngram_size=2  # Reduce repetition
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_text = generated_text[len(request.prompt):].strip()
        
        # Calculate metrics
        inference_time = time.time() - start_time
        output_length = outputs.shape[1] - input_length
        tokens_per_second = output_length / inference_time if inference_time > 0 else 0
        
        return InferenceResponse(
            generated_text=output_text,
            inference_time=inference_time,
            tokens_per_second=tokens_per_second,
            input_tokens=input_length,
            output_tokens=output_length
        )
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not get model info: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "LLM Quantization API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "inference": "/inference",
            "model_info": "/model/info",
            "docs": "/docs"
        },
        "model_status": "loaded" if model is not None else "not loaded"
    }

if __name__ == "__main__":
    # Get model path from environment or command line
    import sys
    if len(sys.argv) > 1:
        os.environ["MODEL_PATH"] = sys.argv[1]
    
    uvicorn.run(
        app,
        host=get_host(),
        port=8000,
        reload=False,  # Don't reload in production
        workers=1      # Single worker for CPU deployment
    )
