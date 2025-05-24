# app.py
import streamlit as st
from dotenv import load_dotenv
from dotenv import load_dotenv
import os
import requests

# Load environment variables from .env
load_dotenv()

# Get API key from .env
api_key = os.getenv("GROQ_API_KEY")

# Make test request to Groq's models endpoint
response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

# Print the response (list of available models)
print(response.json())
from main import query_answer
st.set_page_config(page_title="ASK GEO", layout="centered")
st.title("📚 ASK GEO")
st.markdown("Ask any question related to your uploaded knowledge base!")
user_question = st.text_input("Ask your question:")
if user_question:
    with st.spinner("Thinking..."):
        answer = query_answer(user_question)
        st.success(answer)