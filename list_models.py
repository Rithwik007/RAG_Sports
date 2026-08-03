import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Available models:")
for model in genai.list_models():
    print(f"  - {model.name}")
    print(f"    Supported methods: {model.supported_generation_methods}")
