from setuptools import setup, find_packages
import os

# Read README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "LLM Quantization Pipeline for CPU Deployment"

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="llm-quantization-pipeline",
    version="1.0.0",
    author="Data Science Team",
    author_email="datascience@company.com",
    description="CPU-optimized LLM quantization pipeline for 8GB RAM systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/company/llm-quantization-pipeline",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Extra dependencies for development
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.20.0",
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
        ],
        "monitoring": [
            "wandb>=0.15.0",
            "tensorboard>=2.12.0",
            "mlflow>=2.0.0",
        ]
    },
    
    # Python version requirement
    python_requires=">=3.8,<4.0",
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords
    keywords="llm, quantization, machine-learning, cpu, inference, transformers, gptq, bitsandbytes",
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "llm-quantize=scripts.quantize_model:main",
            "llm-evaluate=scripts.run_evaluation:main",
            "llm-benchmark=scripts.benchmark:main",
            "llm-download=scripts.download_model:main",
            "llm-serve=src.api_server:main",
        ],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/company/llm-quantization-pipeline/issues",
        "Source": "https://github.com/company/llm-quantization-pipeline",
        "Documentation": "https://github.com/company/llm-quantization-pipeline/wiki",
    },
    
    # License
    license="MIT",
    
    # Zip safe
    zip_safe=False,
    
    # Minimum setuptools version
    setup_requires=["setuptools>=45", "wheel"],
)
