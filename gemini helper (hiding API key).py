import streamlit as st
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load the model
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

def summarize_stock(text):
    """Returns a short summary of the stock behavior using Gemini."""
    prompt = (
        "Summarize the following stock analysis in 2–3 simple sentences, "
        "written clearly for a beginner investor:\n\n"
        f"{text}"
    )
    
    response = gemini_model.generate_content(prompt)

    return getattr(response, "text", str(response))
