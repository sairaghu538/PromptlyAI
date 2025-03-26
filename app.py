import streamlit as st
import requests
from transformers import pipeline
import os
from dotenv import load_dotenv

# print(transformers.__version__)

# Load environment variables from the .env file
load_dotenv()

# Access your API keys from environment variables
DEEPAI_API_KEY = os.getenv('DEEPAI_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Ensure that the API keys are loaded properly
if not DEEPAI_API_KEY or not HUGGINGFACE_API_KEY:
    st.error("API keys are not loaded correctly. Please check your .env file.")

# Text generation using Hugging Face (local model with transformers)
def generate_text(prompt):
    try:
        generator = pipeline('text-generation', model='gpt2')
        return generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return "Sorry, there was an error generating text."

# Image generation using DeepAI API
def generate_image_deepai(prompt):
    if not DEEPAI_API_KEY:
        st.error("DeepAI API key is missing.")
        return None
    headers = {'Api-Key': DEEPAI_API_KEY}
    data = {'text': prompt}
    try:
        response = requests.post("https://api.deepai.org/api/text2img", headers=headers, data=data)
        response.raise_for_status()  # Will raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred with the DeepAI API: {e}")
        return None

# Image generation using Hugging Face's Stable Diffusion model (requires API key)
def generate_image_huggingface(prompt):
    if not HUGGINGFACE_API_KEY:
        st.error("Hugging Face API key is missing.")
        return None
    url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    data = {"inputs": prompt}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Will raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred with Hugging Face API: {e}")
        return None

# Streamlit app layout
st.title("Open-Source LLM Chatbot with Text-to-Image Features")
st.write("Ask your question:")

user_input = st.text_input("Ask me anything:")

if user_input:
    # Text generation using Hugging Face GPT-2 model
    generated_text = generate_text(user_input)
    st.write("Generated Text:", generated_text)

    # Image generation with DeepAI or Hugging Face (comment out the one you don't use)
    deepai_image_result = generate_image_deepai(user_input)
    if deepai_image_result and 'output_url' in deepai_image_result:
        st.image(deepai_image_result['output_url'], caption="Generated Image (DeepAI)")

    # Alternatively, if using Hugging Face for Stable Diffusion
    huggingface_image_result = generate_image_huggingface(user_input)
    if huggingface_image_result and 'url' in huggingface_image_result:
        st.image(huggingface_image_result['url'], caption="Generated Image (Hugging Face)")
