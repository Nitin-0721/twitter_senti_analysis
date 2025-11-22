import streamlit as st
import pandas as pd
import spacy
import pickle
import requests
from spacy.lang.en.stop_words import STOP_WORDS

# ----------------------------------------------------------
# LOAD SPACY MODEL (Blank model - works on Streamlit Cloud)
# ----------------------------------------------------------
nlp = spacy.blank("en")

def preprocess(text):
    """Clean tweet text using spaCy"""
    if not isinstance(text, str):
        text = str(text)
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.text.lower() in STOP_WORDS:
            continue
        tokens.append(token.text.lower())
    return " ".join(tokens)

# ----------------------------------------------------------
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# ----------------------------------------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1h8XdP-f-8AiBeL_i9-nVNDB1DWOlyQte"
ENCODER_URL = "https://drive.google.com/uc?export=download&id=19k84TwwLbVh1UUK6SLMBZeZY3t966mmC"

@st.cache_resource
def load_model():
    # Load model
    model_data = requests.get(MODEL_URL).content
    model = pickle.loads(model_data)

    # Load label encoder
    encoder_data = requests.get(ENCODER_URL).content
    label_encoder = pickle.loads(encoder_data)

    return model, label_encoder

model, label_encoder = load_model()

# ----------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------
st.title("Twitter Sentiment Analysis App")
st.write("Enter a tweet below to predict its sentiment.")

tweet = st.text_area("Tweet Text", "")

if st.button("Predict Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter text to analyze.")
    else:
        cleaned = preprocess(tweet)
        pred = model.predict([cleaned])
        sentiment = label_encoder.inverse_transform(pred)[0]

        st.subheader("Prediction:")
        st.success(f"Sentiment: **{sentiment}**")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Machine Learning")
