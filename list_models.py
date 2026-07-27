import google.generativeai as genai
import os

key = os.environ.get("GEMINI_API_KEY")
if not key:
    print("No GEMINI_API_KEY set.")
else:
    genai.configure(api_key=key)
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
