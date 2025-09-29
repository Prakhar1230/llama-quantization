# LLaMA Quantization Pipeline - Docker Image

A containerized LLaMA model quantization pipeline that includes model downloading, quantization, and evaluation capabilities.

## Quick Start

### Pull the Image

```
docker pull prakhar067/llama-quantization:latest
```

## Usage Options

### Option 1: Interactive Container Access

Run the container interactively and execute the pipeline manually:

```
docker run -it --rm --entrypoint /bin/bash prakhar067/llama-quantization:latest
```
Once inside the container, run the complete pipeline:

```
./run_pipeline.sh
```
### Option 2: Direct Pipeline Execution

Run the complete quantization pipeline automatically:

```
docker run -it --rm prakhar067/llama-quantization:latest
```

This will automatically execute the full pipeline including:
- Model downloading (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Model quantization using dynamic method
- Model evaluation with 10 samples
- Interactive pipeline launch