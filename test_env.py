import torch
import openai
import ollama
import azure.ai.inference
import numpy as np
import sys
import os

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"OpenAI version: {openai.__version__}")
print(f"Ollama imported successfully")
print(f"Azure AI Inference imported successfully")
print("\nAll dependencies imported successfully!")
print(f"\nVirtual environment: {os.environ.get('VIRTUAL_ENV', 'Not in a virtual environment')}") 