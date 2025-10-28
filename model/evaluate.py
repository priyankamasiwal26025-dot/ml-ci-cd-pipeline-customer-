import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv(r"data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop the non-numeric ID column (ensure lowercase)
df = data.drop(['customerID'], axis=1)

# Convert categorical columns to numeric (same as training)
df = pd.get_dummies(df, drop_first=True)

# Define X and y (after encoding, Churn becomes 'Churn_Yes')
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model
model = joblib.load('model/Churn_model.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')
