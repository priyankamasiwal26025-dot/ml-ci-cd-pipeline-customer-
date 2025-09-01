import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.dropna()  # drop missing values (same as train.py)

# Prepare features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# One-hot encode categorical variables (must match train.py)
X = pd.get_dummies(X)

# Load the trained model
model = joblib.load('model/Churn_model.pkl')

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y, y_pred))
