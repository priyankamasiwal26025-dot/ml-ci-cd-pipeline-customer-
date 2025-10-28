import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv(r"data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop non-numeric columns that are not features
df = data.drop(['CustomerID'], axis=1)

# Convert categorical variables to numeric
df = pd.get_dummies(df, drop_first=True)

# Preprocess the dataset
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model/Churn_model.pkl')