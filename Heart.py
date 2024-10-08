import streamlit as st
import pandas as pd
import joblib  # For loading the saved model
import numpy as np

# Load the model (ensure the path is correct)
model = joblib.load('your_model_path.joblib')  # Replace with your model's file path

# Function for prediction
def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit App
st.title('Heart Disease Prediction App')

# Create input fields for your features
age = st.number_input('Age', min_value=0, max_value=120)
sex = st.selectbox('Sex', [0, 1])  # Assuming binary encoding: 0 = Female, 1 = Male
cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])  # Modify according to your model's encoding
trestbps = st.number_input('Resting Blood Pressure', min_value=70, max_value=200)
chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])  # 0 = False, 1 = True
restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220)
exang = st.selectbox('Exercise Induced Angina', [0, 1])  # 0 = No, 1 = Yes
oldpeak = st.number_input('Depression Induced by Exercise', min_value=0.0, max_value=6.0)
slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia', [0, 1, 2, 3])  # Modify as per your encoding
target = 1  # Assuming 1 is the positive class for heart disease

# Create a button for prediction
if st.button('Predict'):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal],
        'target': [target]  # Assuming this is part of your input for the model
    })

    # Make prediction
    result = predict(input_data)

    # Display the result
    if result[0] == 1:
        st.write('**Prediction:** Heart Disease Detected')
    else:
        st.write('**Prediction:** No Heart Disease Detected')

# Display the app's information
st.sidebar.header('About')
st.sidebar.text('This app uses a logistic regression model to predict heart disease.')
