import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Title of the dashboard
st.title("Fashion Product Sentiment Analysis Dashboard")

# Load dataset
@st.cache
def load_data():
    data = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
    return data

data = load_data()

# Display raw data
st.subheader("Dataset Preview")
st.write(data.head())

# Display sentiment distribution
st.subheader("Sentiment Distribution")
data['Sentiment'] = data['Rating'].apply(lambda x: 'Positive' if x > 3 else 'Negative' if x < 3 else 'Neutral')
sentiment_counts = data['Sentiment'].value_counts()
fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution")
st.plotly_chart(fig)

# Data Preprocessing for Sentiment Analysis
st.subheader("Training Sentiment Analysis Model")
# Select a subset of columns
df = data[['Review Text', 'Sentiment']].dropna()

# Text Vectorization
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Review Text'])
y = df['Sentiment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Adding User Input for Predictions
st.subheader("Test the Model with a New Review")
user_input = st.text_area("Enter a review:")
if user_input:
    user_input_transformed = vectorizer.transform([user_input])
    prediction = model.predict(user_input_transformed)
    st.write(f"Predicted Sentiment: {prediction[0]}")
