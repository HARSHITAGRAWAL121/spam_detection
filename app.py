import os
os.system("pip install --upgrade nltk")
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function to transform text (preprocessing)
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Keep only alphanumeric words
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Stem the words
    text = [ps.stem(word) for word in text]

    return " ".join(text)  # Join the list into a single string and return

# Load the vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))  # Ensure the correct path to the vectorizer
    model = pickle.load(open('model.pkl', 'rb'))  # Ensure the correct path to the classifier
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()  # Stop execution if files are not found
except Exception as e:
    st.error(f"An error occurred while loading the model or vectorizer: {e}")
    st.stop()  # Stop execution if an error occurs

# Streamlit UI
st.title("Email/SMS Spam Classifier")

# User input
input_sms = st.text_input("Enter SMS")

# Button to predict
if st.button('Predict'):
    if input_sms:
        # 1. Preprocess
        transformed_text = transform_text(input_sms)

        # 2. Vectorize the input text
        vector_input = tfidf.transform([transformed_text])

        # 3. Predict using the model
        result = model.predict(vector_input)[0]  # Get the prediction result

        # 4. Display the result
        if result == 1:
            st.header("SPAM")
        else:
            st.header("NOT SPAM")
    else:
        st.warning("Please enter a message.")
