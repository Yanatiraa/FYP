import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
@st.cache(allow_output_mutation=True)
def load_resources():
    model = load_model("sentiment_rnn_model.h5")  # Load pre-trained model
    with open("tokenizer.pkl", "rb") as file:     # Load Tokenizer
        tokenizer = pickle.load(file)
    return model, tokenizer

model, tokenizer = load_resources()

# Streamlit UI
st.title("Sentiment Analysis with Pre-trained RNN")
st.subheader("Analyze the sentiment of product reviews.")

# User input
user_input = st.text_area("Enter a product review:")

if user_input:
    # Preprocess the input
    sequences = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

    # Predict sentiment
    prediction = model.predict(padded)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    # Display the result
    st.write(f"**Predicted Sentiment:** {sentiment}")
