import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import nltk

# Download NLTK resources
nltk.download('stopwords')

# Preprocessing Function
def preprocess_text(text):
    stemmer = SnowballStemmer('english')
    stop = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop])
    return text

# Load Model and Tokenizer
@st.cache_resource
def load_resources():
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    model = load_model("rnn_sentiment_model.h5")
    return tokenizer, model

# Main Streamlit Application
def main():
    st.title("Sentiment Analysis for Product Reviews")
    st.write("Analyze the sentiment of product reviews based on user input.")

    # User Input for Comment
    comment = st.text_input("Enter your comment about the product:")

    # Spell check the user input
    spell = SpellChecker()
    misspelled_words = [word for word in comment.split() if word not in spell]
    corrected_comment = " ".join([spell.correction(word) for word in comment.split()])

    if misspelled_words:
        st.warning(f"Misspelled words detected: {', '.join(misspelled_words)}")
        st.info(f"Auto-corrected comment: {corrected_comment}")

    # Image Upload for Product
    uploaded_image = st.file_uploader("Upload a product image (optional):", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Product Image", use_column_width=True)

    # Sentiment Analysis
    if st.button("Analyze Sentiment"):
        if corrected_comment:
            tokenizer, model = load_resources()

            # Preprocess and predict
            processed_text = preprocess_text(corrected_comment)
            sequence = tokenizer.texts_to_sequences([processed_text])
            padded_sequence = pad_sequences(sequence, maxlen=100)

            prediction = model.predict(padded_sequence)[0][0]
            sentiment = "Positive" if prediction > 0.5 else "Negative"

            # Display sentiment result
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error("Please enter a valid comment.")

if __name__ == "__main__":
    main()
