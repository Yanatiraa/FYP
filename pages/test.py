# prompt: sentiment analysis of product reviews using RNN

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# Load the dataset
data = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

# Preprocess the data (same as before)
data = data[['Review Text', 'Rating']].dropna()
data['Sentiment'] = data['Rating'].apply(lambda x: 1 if x > 3 else 0)

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Review Text'])
sequences = tokenizer.texts_to_sequences(data['Review Text'])
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
y = data['Sentiment'].values

# Load the saved model and tokenizer
from tensorflow import keras
model = keras.models.load_model("sentiment_rnn_model.h5")
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

def predict_sentiment(review):
    # Preprocess the input review
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Make prediction
    prediction = model.predict(padded_sequence)[0][0]

    # Interpret the prediction
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, prediction


# Example usage
new_review = "This dress is absolutely beautiful!"
sentiment, prob = predict_sentiment(new_review)
print(f"Review: {new_review}")
print(f"Sentiment: {sentiment} (Probability: {prob:.4f})")

new_review = "I hate this product, it's terrible."
sentiment, prob = predict_sentiment(new_review)
print(f"Review: {new_review}")
print(f"Sentiment: {sentiment} (Probability: {prob:.4f})")
