# frontend/streamlit_app.py
"""
Streamlit frontend for the Sentiment-based AI Text Generator.
Run: streamlit run frontend/streamlit_app.py
"""

import streamlit as st
from src.app_core import ai_text_generator

# Page config
st.set_page_config(page_title="Sentiment Text Generator", page_icon="✍️")
st.title("AI Text Generator — Sentiment Aligned")
st.write("Enter a prompt and get a paragraph aligned with the detected sentiment.")

# Cache model function to speed up first load
@st.cache_data
def load_ai_model():
    return ai_text_generator

ai_model = load_ai_model()

# Layout
with st.form("gen_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_area("Enter your prompt/topic:", height=160)
    with col2:
        manual_choice = st.selectbox(
            "Manual sentiment (optional):",
            ["Auto-detect", "positive", "neutral", "negative"]
        )
    length = st.slider("Max length (tokens/approx):", min_value=50, max_value=400, value=150, step=10)
    temp = st.slider("Temperature (creativity):", min_value=0.1, max_value=1.0, value=0.7, step=0.05)
    submit = st.form_submit_button("Generate")

# Handling generation
if submit:
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        manual = None if manual_choice == "Auto-detect" else manual_choice
        with st.spinner("Generating... this may take a few seconds (first run downloads model)..."):
            sentiment, generated = ai_model(prompt, manual_sentiment=manual, length=length, temperature=temp)

        # Display output
        st.success(f"Sentiment detected: **{sentiment}**")
        st.markdown("### Generated Text")
        st.write(generated)
        st.markdown("---")

        # Word count
        word_count = len(generated.split())
        st.info(f"Word count: {word_count}")

        # Copy to clipboard
        st.text_area("Copy the generated text:", value=generated, height=200)
        st.write("Tip: Adjust temperature and length sliders to change style and length.")
