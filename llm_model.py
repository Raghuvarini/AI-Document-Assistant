
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def load_groq_llm_model():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Check Streamlit Secrets.")

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3,
        api_key=api_key
    )

    return llm | StrOutputParser()