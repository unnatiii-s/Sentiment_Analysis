import numpy as np
import pandas as pd
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
import joblib
nltk.download("stopwords")
stop_words = stopwords.words("english")

# Load the pre-trained model
model = joblib.load("sentiment_model.pkl")
# Load the vectorizer
vectorizer = joblib.load("vectorizer.pkl")
#clean the text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]"," ",text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    #tokens shall consists of words that are not in stopwords
    return " ".join(tokens)


st.set_page_config(page_title="Movie Sentiment prediction  App", page_icon=":guardsman:", layout="wide")
st.title("Sentiment Analysis App for Movie Reviews")
st.markdown("Enter a movie ")
user_input = st.text_area("Enter the Review", "Type Here...")
if st.button("Predict Sentiment"):
    cleaned=clean_text(user_input)
    vectorized= vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.success(f"The sentiment of the review is: {sentiment}")