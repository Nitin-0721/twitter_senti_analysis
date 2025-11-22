import streamlit as st
import pandas as pd
import spacy
import pickle

# LOAD SPACY MODEL
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    """Clean tweet text using spaCy"""
    if not isinstance(text, str):
        text = str(text)
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        tokens.append(token.lemma_)
    return " ".join(tokens)

# LOAD TRAINED MODEL + LABEL ENCODER

model = pickle.load(open("model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# STREAMLIT UI

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
st.markdown("Built with using Streamlit & Machine Learning")
