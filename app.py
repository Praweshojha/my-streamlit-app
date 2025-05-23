# app.py
import streamlit as st
from main import query_answer
st.set_page_config(page_title="ASK GEO", layout="centered")
st.title("📚 ASK GEO")
st.markdown("Ask any question related to your uploaded knowledge base!")
user_question = st.text_input("Ask your question:")
if user_question:
    with st.spinner("Thinking..."):
        answer = query_answer(user_question)
        st.success(answer)