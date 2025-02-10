# Credit Card Fraud Detection with Front-End Application
 This repository is for the Neural Networks and Deep Learning Course - Assignment 1, focusing on credit card fraud detection. The project utilizes a machine learning model to predict whether a transaction is fraudulent using a synthetic credit card dataset.

## Features
* Fraud detection model trained on a highly imbalanced dataset, with SMOTE to handle class imbalance.
* Preprocessing pipeline with standard scaling for numerical features and handling any missing values.
* Logistic Regression model for prediction, evaluated with precision, recall, F1-score, and ROC-AUC.
* Streamlit front-end app where users can enter transaction details and get real-time fraud predictions.
* Shows probability scores for both fraud (Class 1) and non-fraud (Class 0).
* Input validation to make sure users enter valid data.
* Trained model saved as a .pkl file for easy reuse.
