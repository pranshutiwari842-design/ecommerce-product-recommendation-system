import pandas as pd
import numpy as np
import pickle
import os

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics import ndcg_score

os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Reviews.csv", nrows=50000)

# Select required columns
df = df[["UserId", "ProductId", "Score", "Summary"]]

# Rename columns
df.columns = ["user_id", "product_id", "rating", "summary"]

# Remove missing values
df.dropna(inplace=True)

# Keep active users and products
user_counts = df["user_id"].value_counts()
product_counts = df["product_id"].value_counts()

df = df[df["user_id"].isin(user_counts[user_counts >= 3].index)]
df = df[df["product_id"].isin(product_counts[product_counts >= 3].index)]

print("Final dataset shape:", df.shape)

# Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user_id", "product_id", "rating"]], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Matrix Factorization Model
model = SVD(n_factors=50, n_epochs=20, random_state=42)
model.fit(trainset)

predictions = model.test(testset)

rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

# Precision and Recall function
def precision_recall_at_k(predictions, k=10, threshold=4):
    user_est_true = {}

    for uid, iid, true_r, est, _ in predictions:
        user_est_true.setdefault(uid, []).append((est, true_r))

    precisions = []
    recalls = []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        top_k = user_ratings[:k]

        relevant = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        recommended_relevant = sum((true_r >= threshold) for (_, true_r) in top_k)

        precision = recommended_relevant / k
        recall = recommended_relevant / relevant if relevant != 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)

precision, recall = precision_recall_at_k(predictions, k=10)

# Save model
with open("models/recommender_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save processed data
df.to_csv("models/processed_data.csv", index=False)

# Save metrics
metrics = {
    "RMSE": rmse,
    "MAE": mae,
    "Precision@10": precision,
    "Recall@10": recall
}

with open("models/metrics.pkl", "wb") as file:
    pickle.dump(metrics, file)

print("Model trained successfully!")
print(metrics)