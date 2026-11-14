import joblib
import os
import pandas as pd
from sklearn.pipeline import Pipeline

MODEL_PATH = os.path.join('models', 'pipeline.joblib')

def load_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def predict_cluster(age: float, annual_income: float, spending_score: float) -> int:
    """Predict the cluster for a new customer.
    Parameters are expected in the same units as the training data:
    - age (years)
    - annual_income (k$)
    - spending_score (1‑100)
    Returns the integer cluster label assigned by KMeans.
    """
    model = load_model()
    # Build a DataFrame with the same feature order used in training
    X = pd.DataFrame([[age, annual_income, spending_score]], columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"])
    scaler = model.named_steps["scaler"]
    kmeans = model.named_steps["kmeans"]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    cluster = kmeans.predict(X_scaled)[0]
    return int(cluster)

if __name__ == '__main__':
    # Simple CLI test
    import argparse
    parser = argparse.ArgumentParser(description='Predict customer segment')
    parser.add_argument('--age', type=float, required=True)
    parser.add_argument('--income', type=float, required=True, help='Annual Income in k$')
    parser.add_argument('--score', type=float, required=True, help='Spending Score (1-100)')
    args = parser.parse_args()
    print(f"Predicted cluster: {predict_cluster(args.age, args.income, args.score)}")

