import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained classification model
with open('classifier.pkl', 'rb') as file:
    model = pickle.load(file)

#Set up the Streamlit app
st.title('Parkinson\'s Disease Classification')

# Function to preprocess and scale the input features
def preprocess_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

# Function to make predictions
def predict(features):
    scaled_features = preprocess_features(features)
    prediction = model.predict(scaled_features)
    return prediction

# Create input fields for the 27 features
# feature_columns = ['feature' + str(i) for i in range(1, 28)]
feature_columns = [
                               'Jitter_local','Jitter_local_absolute','Jitter_rap','Jitter_ppq5','Jitter_ddp',
                               'Shimmer_local','Shimmer_local_dB','Shimmer_apq3','Shimmer_apq5', 'Shimmer_apq11','Shimmer_dda', 
                               'AC','NTH','HTN', 
                               'Median_pitch','Mean_pitch','Standard_deviation','Minimum_pitch','Maximum_pitch',
                               'Number_of_pulses','Number_of_periods','Mean_period','Standard_deviation_of_period',
                               'Fraction_of_locally_unvoiced_frames','Number_of_voice_breaks','Degree_of_voice_breaks', 
]
feature_values = []
for column in feature_columns:
    value = st.number_input(column, step=0.01)
    feature_values.append(value)

# Make prediction when the user clicks the "Predict" button
if st.button('Predict'):
    features = pd.DataFrame([feature_values], columns=feature_columns)
    prediction = predict(features)
    
    # Convert the prediction to a readable label
    if prediction[0] == 0:
        result = 'Healthy'
    else:
        result = 'Parkinson\'s Disease'
    
    # Display the prediction result
    st.write('Prediction:', result)