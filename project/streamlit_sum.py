import streamlit as st
from transformers import pipeline

# Load Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.title("📝 AI Text Summarizer")
st.write("Enter your text, and the AI will generate a summary!")

# User input
text = st.text_area("Paste your text here:")

if st.button("Summarize"):
    if text:
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
        st.subheader("Summary:")
        st.write(summary[0]['summary_text'])
    else:
        st.warning("⚠️ Please enter some text!")
