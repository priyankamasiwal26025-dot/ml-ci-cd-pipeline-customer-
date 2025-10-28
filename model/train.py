import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load the dataset
data = pd.read_csv(r"data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop the non-numeric CustomerID column
df = data.drop(['customerID'], axis=1)  # Note: the actual column name is usually 'customerID' (lowercase)

# Convert categorical variables to numeric
df = pd.get_dummies(df, drop_first=True)

# Define features (X) and target (y)
X = df.drop('Churn_Yes', axis=1)  # After one-hot encoding, Churn becomes 'Churn_Yes'
y = df['Churn_Yes']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ensure model directory exists
os.makedirs('model', exist_ok=True)

# Save the model
joblib.dump(model, 'model/Churn_model.pkl')

print("Model trained and saved successfully!")
