# Import libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Check if model file exists
if not os.path.exists("fraud_model.pkl"):
    st.error("Model file 'fraud_model.pkl' not found. Please ensure it is in the correct directory.")

# Load the preprocessor and model from pickle file
with open("fraud_model.pkl", "rb") as model_file:
    preprocessor, model = pickle.load(model_file)

# Define user input features
st.title("Credit Card Fraud Detection")
st.write("This web application detects fraudulent credit card transactions. Please enter transaction details.")

# Define input fields based on the dataset
# Define input fields based on the dataset
amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=0.01)
time = st.number_input("Transaction Time (Elapsed Seconds)", min_value=1, step=1)
transaction_type = st.selectbox("Transaction Type", ["Online", "POS", "ATM"])
location = st.selectbox("Location", ["UK", "Germany", "Canada", "USA"])
card_type = st.selectbox("Card Type", ["Visa", "MasterCard", "Discover"])
merchant_category = st.selectbox("Merchant Category", ["Travel", "Grocery", "Electronics", "Clothing"])
device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
previous_fraudulent = st.selectbox("Previous Fraudulent Transactions", [0, 1])

# Make prediction
if st.button("Predict Fraud"):
    if amount == 0.0:
        st.error("Transaction Amount must be greater than zero.")
    elif time < 1:
        st.error("Transaction Time must be at least 1 second.")
    elif transaction_type not in ["Online", "POS", "ATM"]:
        st.error("Invalid Transaction Type selected.")
    elif location not in ["UK", "Germany", "Canada", "USA"]:
        st.error("Invalid Location selected.")
    elif card_type not in ["Visa", "MasterCard", "Discover"]:
        st.error("Invalid Card Type selected.")
    elif merchant_category not in ["Travel", "Grocery", "Electronics", "Clothing"]:
        st.error("Invalid Merchant Category selected.")
    elif device_type not in ["Mobile", "Desktop", "Tablet"]:
        st.error("Invalid Device Type selected.")
    elif previous_fraudulent not in [0, 1]:
        st.error("Invalid value for Previous Fraudulent Transactions.")
    else:
        # Create a dataframe with user inputs
        input_data = pd.DataFrame([[amount, time, transaction_type, location, card_type, merchant_category, device_type, previous_fraudulent]], 
                                   columns=["Amount", "Time", "Transaction_Type", "Location", "Card_Type", "Merchant_Category", "Device_Type", "Previous_Fraudulent"])
        
        # Apply preprocessing
        input_processed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_processed)[0]
        prediction_prob = model.predict_proba(input_processed)[0]
        
        # Display results
        st.write(f"### Fraud Prediction: {'Fraudulent' if prediction == 1 else 'Not Fraudulent'}")
        st.write(f"#### Probability of Fraud: {prediction_prob[1]:.4f}")
        st.write(f"#### Probability of Non-Fraud: {prediction_prob[0]:.4f}")