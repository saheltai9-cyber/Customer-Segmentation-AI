import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.clustering import evaluate_clusters, fit_kmeans
from sklearn.pipeline import Pipeline

DATA_RAW_PATH = os.path.join("data", "raw", "Mall_Customers.csv")
MODEL_PATH = os.path.join("models", "pipeline.joblib")
REPORTS_DIR = "reports"

def main():
    # 1. Load & validate data
    df = load_data(DATA_RAW_PATH)
    print(f"Dataset loaded – {df.shape[0]} records, columns: {list(df.columns)}")

    # 2. Preprocess
    X_scaled, scaler = preprocess(df)

    # 3. Evaluate cluster numbers 2-10
    eval_df = evaluate_clusters(X_scaled, range(2, 11))
    eval_df.to_csv(os.path.join(REPORTS_DIR, "cluster_evaluation.csv"), index=False)
    print("Cluster evaluation saved.")

    # 4. Choose cluster count – highest silhouette score (break ties with lower DB)
    best_row = eval_df.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]
    best_k = int(best_row["k"])
    print(f"Selected k={best_k} based on silhouette={best_row['silhouette']:.4f}, DB={best_row['davies_bouldin']:.4f}")

    # 5. Fit final model
    final_model = fit_kmeans(X_scaled, best_k)
    # Build pipeline: scaler then model
    pipeline = joblib.load(os.path.join("models", "pipeline_placeholder.pkl")) if os.path.exists(os.path.join("models", "pipeline_placeholder.pkl")) else None
    # Instead of loading placeholder, just create new pipeline object
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([("scaler", scaler), ("kmeans", final_model)])
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Pipeline saved to {MODEL_PATH}")

    # 6. Generate simple visualizations (Elbow chart)
    plt.figure()
    plt.plot(eval_df["k"], eval_df["inertia"], marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(eval_df["k"])
    plt.grid(True)
    plt.savefig(os.path.join(REPORTS_DIR, "elbow.png"))
    plt.close()
    print("Elbow chart saved.")

if __name__ == "__main__":
    main()
