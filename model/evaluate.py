import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv(r"data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocess the dataset
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model
model = joblib.load('model/Churn_model.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')