import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Streamlit page layout
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Check if model file exists
if not os.path.exists("fraud_model.pkl"):
    st.error("Model file 'fraud_model.pkl' not found. Please ensure it is in the correct directory.")
else:
    # Load the saved model, preprocessor and best threshold
    with open("fraud_model.pkl", "rb") as file:
        preprocessor, model, best_threshold = pickle.load(file)

    # Sidebar for input fields
    st.sidebar.title("Enter Transaction Details")
    amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, step=0.01)
    time = st.sidebar.number_input("Transaction Time (Elapsed Seconds)", min_value=1, step=1)
    transaction_type = st.sidebar.selectbox("Transaction Type", ["Online", "POS", "ATM"])
    location = st.sidebar.selectbox("Location", ["UK", "Germany", "Canada", "USA", "India"])
    card_type = st.sidebar.selectbox("Card Type", ["Amex", "Visa", "MasterCard", "Discover"])
    merchant_category = st.sidebar.selectbox("Merchant Category", ["Electronics", "Food", "Grocery", "Retail", "Travel"])
    device_type = st.sidebar.selectbox("Device Type", ["Mobile", "Desktop", "Tablet", "Laptop"])
    previous_fraudulent = st.sidebar.selectbox("Previous Fraudulent Transactions", [0, 1])

    # Input validation and Prediction Button
    if st.sidebar.button("Predict Fraud"):
        if amount == 0.0:
            st.sidebar.error("Transaction Amount must be greater than zero.")
        elif time < 1:
            st.sidebar.error("Transaction Time must be at least 1 second.")
        else:
            # Create a dataframe with user inputs
            input_data = pd.DataFrame([[amount, time, transaction_type, location, card_type, merchant_category, device_type, previous_fraudulent]], 
                                    columns=["Amount", "Time", "Transaction_Type", "Location", "Card_Type", "Merchant_Category", "Device_Type", "Previous_Fraudulent"])
            
            # Apply preprocessing
            input_processed = preprocessor.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_processed)[0]
            prediction_prob = model.predict_proba(input_processed)[0]
            
            # Main Display Section
            st.title("Fraud Detection Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Prediction Result")
                st.markdown(f"## {'🛑 Fraudulent' if prediction == 1 else '✅ Not Fraudulent'}")
                st.markdown(f"**Probability of Fraud:** {prediction_prob[1]:.4f}")
                st.markdown(f"**Probability of Non-Fraud:** {prediction_prob[0]:.4f}")
                
                # Gauge-like representation of fraud probability
                st.progress(int(prediction_prob[1] * 100))
            
            with col2:
                # Visualization - Probability Bar Chart
                st.markdown("### Fraud Probability Distribution")
                fig, ax = plt.subplots()
                sns.barplot(x=["Non-Fraud", "Fraud"], y=[prediction_prob[0], prediction_prob[1]], ax=ax, palette=["blue", "red"])
                ax.set_ylabel("Probability")
                st.pyplot(fig)