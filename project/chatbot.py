import os
from dotenv import load_dotenv
from transformers import pipeline

# Load API keys
load_dotenv()
huggingface_key = os.getenv("HUGGINGFACE_API_KEY")

# Load chatbot model
chatbot = pipeline("text-generation", model="gpt2")

def chat_with_hf(prompt):
    response = chatbot(prompt, max_length=150, do_sample=True)
    return response[0]['generated_text']

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        response = chat_with_hf(user_input)
        print("Chatbot:", response)
