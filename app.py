
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Model
with open('ann_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Selected Features
selected_features = ['mean radius', 'mean perimeter', 'mean area', 'mean concavity', 'mean concave points', 'worst radius', 'worst perimeter', 'worst area', 'worst concavity', 'worst concave points']

# App Title
st.title("Breast Cancer Prediction App")
st.write("This app has been developed by LAURA OSORIO BERMUDEZ -  C0917325. This app uses a neural network to predict if a tumor is malignant or benign.")

# Input Features with Sliders
st.sidebar.header("Input Features")
user_input = {}
for feature in selected_features:
    user_input[feature] = st.sidebar.slider(
        f"Enter value for: {feature}",
        min_value=-5.0,
        max_value=5.0,
        step=0.1
    )

# Predict Button
if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)
    result = "Malignant" if prediction[0] == 1 else "Benign"
    st.write(f"The predicted tumor type is: {result}")
