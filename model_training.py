# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
import pickle

# Load data
df = pd.read_csv("dataset/credit_card_fraud_dataset.csv")

# Handle missing values
missing_values = df.isnull().sum()
print(missing_values)

# Drop Transaction_ID
df.drop(columns=["Transaction_ID"], inplace=True)

# Split features and target
X = df.drop(columns=["Fraudulent"])
y = df["Fraudulent"]

# Check for class imbalance
class_distribution = y.value_counts()
print(class_distribution)

# Identify numerical columns
num_features = ["Amount", "Time", "Previous_Fraudulent"]
num_transformer = StandardScaler()

# Identify categorical columns
cat_features = ["Transaction_Type", "Location", "Card_Type", "Merchant_Category", "Device_Type"]
cat_transformer = OneHotEncoder(handle_unknown='ignore')

# Preprocessing numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)

# Apply preprocessing to features
X_processed = preprocessor.fit_transform(X)

# Print transformed features
print("Processed feature names:", preprocessor.get_feature_names_out())

# Use SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

# Print resampled class distribution
class_distribution_resampled = y_resampled.value_counts()
print(class_distribution_resampled)

# Train Test Split 80-20
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
print("Classification Report: \n", classification_report(y_test, y_pred))
print("ROC AUC Score: ", roc_auc_score(y_test, y_pred_proba))

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, marker='.', label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# Save preprocessor and model
with open("fraud_model.pkl", "wb") as file:
    pickle.dump((preprocessor, model), file)

print("Model training complete!")