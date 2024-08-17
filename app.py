import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model('language_identification_model.h5')

# Load the label encoder
languages = ['Bengali', 'Gujarati', 'Hindi', 'Kannada', 'Malayalam', 'Marathi', 'Odia', 'Punjabi', 'Tamil', 'Telugu']
label_encoder = LabelEncoder()
label_encoder.fit(languages)

# Function to preprocess the audio file
def preprocess_audio(file):
    # Load the audio file
    audio, sample_rate = librosa.load(file, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Reshape the features for the model input
    mfccs_mean = mfccs_mean.reshape(1, -1)
    
    return mfccs_mean

# Streamlit app
st.title("Language Identification from Audio")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Preprocess the audio file
    features = preprocess_audio(uploaded_file)
    
    # Predict the language
    predictions = model.predict(features)
    predicted_label = np.argmax(predictions, axis=1)
    predicted_language = label_encoder.inverse_transform(predicted_label)[0]
    
    # Display the predicted language
    st.write(f"**The language detected in this audio is: {predicted_language}**")
