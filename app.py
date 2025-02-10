# Import libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
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
        
        # Visualization - Probability Bar Chart
        fig, ax = plt.subplots()
        sns.barplot(x=["Non-Fraud", "Fraud"], y=[prediction_prob[0], prediction_prob[1]], ax=ax, palette=["blue", "red"])
        ax.set_ylabel("Probability")
        ax.set_title("Fraud Probability Distribution")
        st.pyplot(fig)
        
        # Generate evaluation metrics
        y_test = np.array([0, 1])  # Dummy test values for visualization
        y_pred_prob = np.array([prediction_prob[0], prediction_prob[1]])
        
        # Compute ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC Curve
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
        
        # Display classification report (for reference)
        report = classification_report([0, 1], [int(prediction_prob[0] < prediction_prob[1]), int(prediction_prob[1] > prediction_prob[0])], output_dict=True)
        st.write("### Classification Report")
        st.json(report)