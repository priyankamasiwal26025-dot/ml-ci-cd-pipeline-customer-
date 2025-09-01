import unittest
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

class TestModel(unittest.TestCase):
    def test_model_training(self):
        # Load dataset
        df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv").dropna()
        X = pd.get_dummies(df.drop('Churn', axis=1))
        y = df['Churn']

        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Ensure model is fitted and can make predictions
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))  # predictions length matches data length

    def test_model_file_exists(self):
        # Check if model file exists after training
        try:
            model = joblib.load('model/Churn_model.pkl')
            self.assertIsNotNone(model)
        except FileNotFoundError:
            self.fail("❌ Model file not found. Run train.py before testing.")

if __name__ == "__main__":
    unittest.main()
