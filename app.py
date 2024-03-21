import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to tokenize text using spaCy
def spacy_tokenizer(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return tokens

# Function to preprocess input text
def preprocess_text(text):
    tokens = spacy_tokenizer(text)
    return ' '.join(tokens)

# Load label encoder
label_encoder = joblib.load('notebook\label_encoder.pkl')

# Load saved model
model = load_model('notebook\hate_speech_detection_model.h5')

# Streamlit app
st.title('Hate Speech Detection App')

# Input text box for user input
user_input = st.text_input('Enter text:')

# Preprocess user input
if user_input:
    processed_input = preprocess_text(user_input)

    # Tokenize input text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([processed_input])

    # Save tokenizer
    joblib.dump(tokenizer, 'tokenizer.pkl')

    sequences = tokenizer.texts_to_sequences([processed_input])
    padded_sequence = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

    # Predict
    prediction = model.predict(padded_sequence).argmax(axis=1)
    prediction_label = label_encoder.inverse_transform(prediction)[0]

    # Display prediction
    st.write('Prediction:', prediction_label)
