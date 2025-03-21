import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Configuration
PROJECT_ROOT = "/Users/owaiskamdar/Desktop/resume_optimizer/lllm-resume-optimizer"
DATA_PATH = os.path.join(PROJECT_ROOT, "data/outputs/matched_resumes_jobs_withfeatures.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load Dataset
df_gmm = pd.read_csv(DATA_PATH)

# Feature Scaling
scaler_features = MinMaxScaler()
X_scaled = scaler_features.fit_transform(df_gmm[["tfidf_cosine", "jaccard", "bert_similarity", "ngram_overlap"]])

# Train GMM
num_components = 5
gmm = GaussianMixture(n_components=num_components, random_state=42)
gmm.fit(X_scaled)

# Compute GMM Scores
probabilities = gmm.predict_proba(X_scaled)
gmm_scores = np.dot(probabilities, np.linspace(0, 100, num_components))

# Normalize GMM Scores
scaler_gmm = MinMaxScaler(feature_range=(0, 100))
gmm_scores_scaled = scaler_gmm.fit_transform(gmm_scores.reshape(-1, 1)).flatten()
df_gmm["gmm_match_score"] = gmm_scores_scaled

# Plot Score Distribution
plt.figure(figsize=(10, 5))
plt.hist(gmm_scores_scaled, bins=20, edgecolor="black", alpha=0.5)
plt.title("GMM Match Score Distribution")
plt.xlabel("Match Score")
plt.ylabel("Frequency")
plt.show()

# Save GMM & Scalers
with open(os.path.join(MODEL_DIR, "gmm_model.pkl"), "wb") as f:
    pickle.dump(gmm, f)

with open(os.path.join(MODEL_DIR, "scaler_gmm.pkl"), "wb") as f:
    pickle.dump(scaler_gmm, f)

with open(os.path.join(MODEL_DIR, "scaler_features.pkl"), "wb") as f:
    pickle.dump(scaler_features, f)

# Train ML Models
X = scaler_features.transform(df_gmm[["tfidf_cosine", "jaccard", "bert_similarity", "ngram_overlap"]])
y = df_gmm["gmm_match_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "R² Score": r2}

    with open(os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}_model.pkl"), "wb") as f:
        pickle.dump(model, f)

# Print Evaluation Results
print("\nModel Performance on GMM-Based Match Scores:")
for model, metrics in results.items():
    print(f"{model}: MAE = {metrics['MAE']:.3f}, R² = {metrics['R² Score']:.3f}")


# Save updated DataFrame with GMM score
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data/outputs/matched_resumes_jobs_withgmm.csv")
df_gmm.to_csv(OUTPUT_CSV, index=False)
print(f"Updated dataset saved to: {OUTPUT_CSV}")
