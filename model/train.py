import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop rows with missing values (optional but recommended)
df = df.dropna()

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Convert categorical variables into dummy/one-hot encoded variables
X = pd.get_dummies(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model/Churn_model.pkl')

print("✅ Model training complete! Saved to model/Churn_model.pkl")
