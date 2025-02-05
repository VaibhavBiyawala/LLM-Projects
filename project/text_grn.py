import os
from dotenv import load_dotenv
from transformers import pipeline


load_dotenv()
# openai_key = os.getenv("OPENAI_API_KEY")
huggingface_key = os.getenv("HUGGINGFACE_API_KEY")
# groq_key = os.getenv("GROQ_API_KEY")

# Load text generation model
generator = pipeline("text-generation", model="gpt2")

# Generate text
prompt = input("Enter a prompt: ")
response = generator(prompt, max_length=150)
print(response[0]['generated_text'])