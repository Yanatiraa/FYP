# prompt: Code for run in streamlit which is about sentiment analysis of product reviews

import streamlit as st
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
import pickle

# Load the saved model and tokenizer
model = models.load_model("sentiment_rnn_model.h5")
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

max_length = 100

def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, prediction

# Streamlit app
st.title("Product Review Sentiment Analysis")

review_text = st.text_area("Enter product review:", "This product is amazing!")

if st.button("Analyze"):
    sentiment, probability = predict_sentiment(review_text)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Probability: {probability:.4f}")
