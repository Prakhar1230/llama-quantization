# #!/usr/bin/env python3

# import os
# import sys
# import subprocess
# import time
# import json
# import requests
# from datetime import datetime
# from typing import Optional

# sys.path.append('/app/src')
# sys.path.append('/app')

# class InteractiveLLMPipeline:
#     def __init__(self):
#         self.api_process = None
#         self.model_original_path = "/app/models/original"
#         self.model_quantized_path = "/app/models/quantized"

#     def print_banner(self):
#         banner = """
# ==============================================
# >> LLM QUANTIZATION INTERACTIVE PIPELINE <<
# >> Docker-Based Quantization with Comparison <<
# ==============================================

# Welcome to the Interactive LLM Quantization Suite!
# This pipeline will guide you through quantization and testing.
#         """
#         print(banner)

#     def check_prerequisites(self) -> bool:
#         print("\n-- Checking System Prerequisites --")
#         print("---------------------------------")

#         original_exists = os.path.exists(f"{self.model_original_path}/config.json")
#         quantized_exists = os.path.exists(f"{self.model_quantized_path}/quantized_model.pt")

#         print(f"--> Original Model:  {'Found' if original_exists else 'Missing'}")
#         print(f"--> Quantized Model: {'Found' if quantized_exists else 'Missing'}")

#         if not original_exists or not quantized_exists:
#             print("\n-> Models not found. Please ensure the models directory is mounted correctly.")
#             print("   Example: docker run -v $(pwd)/models:/app/models ...")
#             return False

#         try:
#             import torch
#             import transformers
#             import fastapi
#             print("--> Python dependencies: Found")
#         except ImportError as e:
#             print(f"--> Missing dependencies: {e}")
#             return False

#         return True

#     def show_quantization_info(self):
#         print("\n-- Quantization Information --")
#         print("------------------------------")
#         try:
#             print("--> Method Used: DYNAMIC Quantization (PyTorch INT8)")
            
#             orig_size = sum(f.stat().st_size for f in Path(self.model_original_path).glob('**/*') if f.is_file()) / (1024**2)
#             quant_size = os.path.getsize(f"{self.model_quantized_path}/quantized_model.pt") / (1024**2)
            
#             if orig_size > 0 and quant_size > 0:
#                 print(f"--> Original Size:    {orig_size:.1f} MB")
#                 print(f"--> Quantized Size:   {quant_size:.1f} MB")
#                 print(f"--> Size Reduction:   {((orig_size - quant_size) / orig_size * 100):.1f}%")
#                 print(f"--> Compression Ratio: {orig_size / quant_size:.1f}x")

#         except Exception as e:
#             print(f"--> Could not calculate quantization details: {e}")

#     def start_api_server(self) -> bool:
#         print("\n-- Starting API Server --")
#         print("---------------------------")
#         try:
#             os.environ["MODEL_PATH"] = self.model_quantized_path
#             print("--> Starting FastAPI server in the background...")
#             self.api_process = subprocess.Popen(
#                 [sys.executable, "/app/src/api_server.py"],
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True
#             )

#             print("--> Waiting for API to become available...")
#             for i in range(30):
#                 try:
#                     response = requests.get("http://localhost:8000/health", timeout=2)
#                     if response.status_code == 200:
#                         print("--> API server is ready!")
#                         return True
#                 except requests.exceptions.ConnectionError:
#                     pass
#                 time.sleep(1)
#                 print(f"    Waiting... ({i+1}/30)")

#             print("\n--> API server failed to start. Check server logs.")
#             stdout, stderr = self.api_process.communicate()
#             print("\n--- API Server STDOUT ---\n", stdout)
#             print("\n--- API Server STDERR ---\n", stderr)
#             return False
#         except Exception as e:
#             print(f"--> Error starting API server: {e}")
#             return False

#     def run_model_comparison(self):
#         print("\n-- Running Model Comparison --")
#         try:
#             result = subprocess.run(
#                 [sys.executable, "/app/compare_models.py"],
#                 capture_output=True, text=True, timeout=600
#             )
#             if result.returncode == 0:
#                 print("--> Comparison completed. Full report is in model_comparison_results.json")
#                 print("\n--- Comparison Summary ---")
#                 print(result.stdout)
#             else:
#                 print("--> Comparison failed:")
#                 print(result.stderr)
#         except Exception as e:
#             print(f"--> Error during comparison: {e}")

#     def interactive_testing(self):
#         print("\n-- Interactive Prompt Testing --")
#         while True:
#             prompt = input("--> Enter prompt (or 'back'): ").strip()
#             if prompt.lower() == 'back':
#                 break
#             if not prompt:
#                 continue

#             result = self.test_api_prompt(prompt)
#             if result:
#                 print(f"\nResponse:\n{result['generated_text']}")
#                 print(f"\n---> Time: {result['inference_time']:.2f}s, Speed: {result['tokens_per_second']:.1f} tok/s")
#             else:
#                 print("--> Failed to generate response from API.")

#     def test_predefined_prompts(self):
#         prompts = [
#             "What is artificial intelligence?",
#             "Explain machine learning in simple terms.",
#             "How does quantization reduce model size?"
#         ]
#         print("\n-- Testing Predefined Prompts --")
#         for i, p in enumerate(prompts, 1):
#             print(f"\n--> Test {i}: {p}")
#             result = self.test_api_prompt(p, max_length=80)
#             if result:
#                 print(f"    Success: {result['inference_time']:.2f}s, {result['tokens_per_second']:.1f} tok/s")
#             else:
#                 print("    Failed.")

#     def test_api_prompt(self, prompt: str, max_length: int = 100) -> Optional[dict]:
#         try:
#             payload = {"prompt": prompt, "max_length": max_length, "temperature": 0.7}
#             response = requests.post("http://localhost:8000/inference", json=payload, timeout=30)
#             return response.json() if response.status_code == 200 else None
#         except requests.exceptions.RequestException:
#             print("--> API request failed. Is the server running?")
#             return None

#     def show_main_menu(self):
#         print("\n------------------ MAIN MENU ------------------")
#         print("1. Run Complete Model Comparison")
#         print("2. Interactive Prompt Testing")
#         print("3. Test Predefined Prompts")
#         print("4. API Status Check")
#         print("5. Exit")
#         print("---------------------------------------------")

#     def check_api_status(self):
#         print("\n-- API Status Check --")
#         try:
#             response = requests.get("http://localhost:8000/health", timeout=2)
#             if response.status_code == 200:
#                 print("--> API Status: Online")
#             else:
#                 print(f"--> API Status: Error (Code: {response.status_code})")
#         except requests.exceptions.RequestException:
#             print("--> API Status: Offline")

#     def cleanup(self):
#         print("\n-- Cleaning up and exiting --")
#         if self.api_process and self.api_process.poll() is None:
#             print("--> Stopping API server...")
#             self.api_process.terminate()
#             self.api_process.wait()
#         print("--> Cleanup complete. Goodbye.")

#     def run(self):
#         self.print_banner()
#         if not self.check_prerequisites():
#             return
#         self.show_quantization_info()
#         if not self.start_api_server():
#             return

#         try:
#             while True:
#                 self.show_main_menu()
#                 choice = input("--> Select option (1-5): ").strip()
#                 if choice == '1':
#                     self.run_model_comparison()
#                 elif choice == '2':
#                     self.interactive_testing()
#                 elif choice == '3':
#                     self.test_predefined_prompts()
#                 elif choice == '4':
#                     self.check_api_status()
#                 elif choice == '5':
#                     break
#                 else:
#                     print("--> Invalid option. Please try again.")
#         finally:
#             self.cleanup()

# if __name__ == "__main__":
#     pipeline = InteractiveLLMPipeline()
#     pipeline.run()




#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional

class InteractiveLLMPipeline:
    def __init__(self):
        self.api_process = None
        self.is_docker = self.detect_docker_environment()
        
        # Auto-detect paths based on environment
        if self.is_docker:
            # Docker paths
            self.model_original_path = "/app/models/original"
            self.model_quantized_path = "/app/models/quantized"
            self.src_path = "/app/src"
            self.script_path = "/app"
            sys.path.extend(["/app/src", "/app"])
        else:
            # Local environment paths
            self.base_path = Path(__file__).parent.absolute()
            self.model_original_path = str(self.base_path / "models" / "original")
            self.model_quantized_path = str(self.base_path / "models" / "quantized")
            self.src_path = str(self.base_path / "src")
            self.script_path = str(self.base_path)
            sys.path.extend([self.src_path, str(self.base_path)])

    def detect_docker_environment(self) -> bool:
        """Detect if running inside Docker container."""
        try:
            # Method 1: Check for .dockerenv file
            if os.path.exists('/.dockerenv'):
                return True
            
            # Method 2: Check /proc/self/cgroup for docker
            if os.path.exists('/proc/self/cgroup'):
                with open('/proc/self/cgroup', 'r') as f:
                    return 'docker' in f.read()
            
            # Method 3: Check environment variable
            if os.environ.get('DOCKER_CONTAINER', False):
                return True
                
            return False
        except Exception:
            return False

    def print_banner(self):
        env_type = "Docker Container" if self.is_docker else "Local Environment"
        banner = f"""
==============================================
>> LLM QUANTIZATION INTERACTIVE PIPELINE <<
>> Running in: {env_type} <<
==============================================

Welcome to the Interactive LLM Quantization Suite!
This pipeline will guide you through quantization and testing.
        """
        print(banner)

    def check_prerequisites(self) -> bool:
        print(f"\n-- Checking System Prerequisites ({('Docker' if self.is_docker else 'Local')}) --")
        print("-" * 60)

        original_exists = os.path.exists(os.path.join(self.model_original_path, "config.json"))
        quantized_exists = os.path.exists(os.path.join(self.model_quantized_path, "quantized_model.pt"))

        print(f"--> Environment: {'Docker Container' if self.is_docker else 'Local System'}")
        print(f"--> Original Model Path:  {self.model_original_path}")
        print(f"--> Quantized Model Path: {self.model_quantized_path}")
        print(f"--> Original Model:  {'Found' if original_exists else 'Missing'}")
        print(f"--> Quantized Model: {'Found' if quantized_exists else 'Missing'}")

        if not original_exists or not quantized_exists:
            print("\n-> Models not found. Please ensure models are available:")
            if self.is_docker:
                print("   For Docker: docker run -v $(pwd)/models:/app/models ...")
            else:
                print(f"   For Local: Ensure models are in {self.base_path}/models/")
            return False

        try:
            import torch
            import transformers
            import fastapi
            print("--> Python dependencies: Found")
        except ImportError as e:
            print(f"--> Missing dependencies: {e}")
            return False

        return True

    def show_quantization_info(self):
        print("\n-- Quantization Information --")
        print("------------------------------")
        try:
            print("--> Method Used: DYNAMIC Quantization (PyTorch INT8)")
            
            # Calculate sizes
            orig_size = sum(f.stat().st_size for f in Path(self.model_original_path).glob('**/*') if f.is_file()) / (1024**2)
            
            quant_file = Path(self.model_quantized_path) / "quantized_model.pt"
            if quant_file.exists():
                quant_size = quant_file.stat().st_size / (1024**2)
                
                if orig_size > 0 and quant_size > 0:
                    print(f"--> Original Size:    {orig_size:.1f} MB")
                    print(f"--> Quantized Size:   {quant_size:.1f} MB")
                    print(f"--> Size Reduction:   {((orig_size - quant_size) / orig_size * 100):.1f}%")
                    print(f"--> Compression Ratio: {orig_size / quant_size:.1f}x")
            else:
                print("--> Quantized model file not found")

        except Exception as e:
            print(f"--> Could not calculate quantization details: {e}")

    def start_api_server(self) -> bool:
        print("\n-- Starting API Server --")
        print("---------------------------")
        try:
            os.environ["MODEL_PATH"] = self.model_quantized_path
            print("--> Starting FastAPI server in the background...")
            
            # Use appropriate Python executable and paths
            api_script = os.path.join(self.src_path, "api_server.py")
            if not os.path.exists(api_script):
                print(f"--> API script not found at: {api_script}")
                return False
                
            self.api_process = subprocess.Popen(
                [sys.executable, api_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            print("--> Waiting for API to become available...")
            for i in range(300):
                try:
                    response = requests.get("http://localhost:8000/health", timeout=2)
                    if response.status_code == 200:
                        print("--> API server is ready!")
                        return True
                except requests.exceptions.ConnectionError:
                    pass
                time.sleep(1)
                print(f"    Waiting... ({i+1}/300)")

            print("\n--> API server failed to start. Check server logs.")
            stdout, stderr = self.api_process.communicate()
            print("\n--- API Server STDOUT ---")
            print(stdout)
            print("\n--- API Server STDERR ---") 
            print(stderr)
            return False
        except Exception as e:
            print(f"--> Error starting API server: {e}")
            return False

    def run_model_comparison(self):
        print("\n-- Running Model Comparison --")
        try:
            compare_script = os.path.join(self.script_path, "compare_models.py")
            if not os.path.exists(compare_script):
                print(f"--> compare_models.py not found at: {compare_script}")
                return
            result = subprocess.run(
                [sys.executable, compare_script, 
                 "--original_path", self.model_original_path,
                 "--quantized_path", self.model_quantized_path],
                capture_output=True, text=True, timeout=6000
            )
            if result.returncode == 0:
                print("--> Comparison completed. Full report is in model_comparison_results.json")
                print("\n--- Comparison Summary ---")
                print(result.stdout)
            else:
                print("--> Comparison failed:")
                print(result.stderr)
        except Exception as e:
            print(f"--> Error during comparison: {e}")

    def interactive_testing(self):
        print("\n-- Interactive Prompt Testing --")
        while True:
            prompt = input("--> Enter prompt (or 'back'): ").strip()
            if prompt.lower() == 'back':
                break
            if not prompt:
                continue

            result = self.test_api_prompt(prompt)
            if result:
                print(f"\nResponse:\n{result['generated_text']}")
                print(f"\n---> Time: {result['inference_time']:.2f}s, Speed: {result['tokens_per_second']:.1f} tok/s")
            else:
                print("--> Failed to generate response from API.")

    def test_predefined_prompts(self):
        prompts = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "How does quantization reduce model size?"
        ]
        print("\n-- Testing Predefined Prompts --")
        for i, p in enumerate(prompts, 1):
            print(f"\n--> Test {i}: {p}")
            result = self.test_api_prompt(p, max_length=80)
            if result:
                print(f"    Success: {result['inference_time']:.2f}s, {result['tokens_per_second']:.1f} tok/s")
            else:
                print("    Failed.")

    def test_api_prompt(self, prompt: str, max_length: int = 100) -> Optional[dict]:
        try:
            payload = {"prompt": prompt, "max_length": max_length, "temperature": 0.7}
            response = requests.post("http://localhost:8000/inference", json=payload, timeout=30)
            return response.json() if response.status_code == 200 else None
        except requests.exceptions.RequestException:
            print("--> API request failed. Is the server running?")
            return None

    def show_main_menu(self):
        print("\n------------------ MAIN MENU ------------------")
        print("1. Run Complete Model Comparison")
        print("2. Interactive Prompt Testing")
        print("3. Test Predefined Prompts")
        print("4. API Status Check")
        print("5. Exit")
        print("---------------------------------------------")

    def check_api_status(self):
        print("\n-- API Status Check --")
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("--> API Status: Online")
                health_data = response.json()
                print(f"--> Model Loaded: {'Yes' if health_data.get('model_loaded') else 'No'}")
                print(f"--> Model Path: {health_data.get('model_path', 'Unknown')}")
            else:
                print(f"--> API Status: Error (Code: {response.status_code})")
        except requests.exceptions.RequestException:
            print("--> API Status: Offline")

    def cleanup(self):
        print("\n-- Cleaning up and exiting --")
        if self.api_process and self.api_process.poll() is None:
            print("--> Stopping API server...")
            self.api_process.terminate()
            self.api_process.wait()
        print("--> Cleanup complete. Goodbye.")

    def run(self):
        self.print_banner()
        if not self.check_prerequisites():
            return
        self.show_quantization_info()
        if not self.start_api_server():
            return

        try:
            while True:
                self.show_main_menu()
                choice = input("--> Select option (1-5): ").strip()
                if choice == '1':
                    self.run_model_comparison()
                elif choice == '2':
                    self.interactive_testing()
                elif choice == '3':
                    self.test_predefined_prompts()
                elif choice == '4':
                    self.check_api_status()
                elif choice == '5':
                    break
                else:
                    print("--> Invalid option. Please try again.")
        finally:
            self.cleanup()

if __name__ == "__main__":
    pipeline = InteractiveLLMPipeline()
    pipeline.run()
