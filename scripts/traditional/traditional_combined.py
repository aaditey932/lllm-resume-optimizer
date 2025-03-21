#!/usr/bin/env python3
"""
Script for traditional resume-job matching pipeline that combines
text processing and model training in a single workflow.

This script:
1. Preprocesses text
2. Computes similarity features
3. Trains a GMM model to cluster similarity scores
4. Trains ML models (Random Forest and XGBoost)
5. Saves all trained models for later use
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import torch
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Configuration

# Set paths
PROJECT_ROOT = "/Users/owaiskamdar/Desktop/resume_optimizer/lllm-resume-optimizer"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_PATH = os.path.join(DATA_DIR, "outputs/matched_resumes_jobs.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "outputs/matched_resumes_jobs_withfeatures.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "outputs/matched_resumes_jobs_withgmm.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Create directories if they don't exist
os.makedirs(os.path.join(DATA_DIR, "outputs"), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Text Processing Functions

def download_nltk_resources():
    """Download necessary NLTK resources."""
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)

def remove_markup(text):
    """Remove markdown formatting from text."""
    text = str(text)
    text = re.sub(r'\*\*[^*]+\*\*', ' ', text)  # Remove markdown bold
    text = re.sub(r'#\S+', ' ', text)  # Remove hashtags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def preprocess_text(text):
    """Apply text preprocessing steps."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(token) for token in text.split() if token not in stop_words]
    return " ".join(tokens)

def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts."""
    set1, set2 = set(text1.split()), set(text2.split())
    return len(set1 & set2) / len(set1 | set2) if set1 and set2 else 0

def ngram_overlap(text1, text2, ngram_range=(1,3)):
    """Calculate n-gram overlap between two texts."""
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    ngrams1 = set(vectorizer.fit([text1]).get_feature_names_out())
    ngrams2 = set(vectorizer.fit([text2]).get_feature_names_out())
    return len(ngrams1 & ngrams2) / max(len(ngrams1 | ngrams2), 1)

# Processing Pipeline

def process_data():
    """Process the resume and job data to compute similarity features."""
    print("Starting text processing pipeline...")
    
    # Load dataset
    print(f"Loading data from {INPUT_PATH}")
    try:
        final_df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_PATH}")
        print("Make sure the matched resume and job data is available.")
        sys.exit(1)
    
    print(f"Loaded {len(final_df)} records")
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Preprocess text
    print("Preprocessing text...")
    final_df["job_text_clean"] = final_df["job_description"].apply(
        lambda x: preprocess_text(remove_markup(str(x)))
    )
    final_df["resume_text_clean"] = final_df["resume_text"].apply(
        lambda x: preprocess_text(remove_markup(str(x)))
    )
    
    # Compute similarity features
    print("Computing TF-IDF similarity...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=15, max_df=0.9)
    tfidf_job = tfidf_vectorizer.fit_transform(final_df["job_text_clean"])
    tfidf_resume = tfidf_vectorizer.transform(final_df["resume_text_clean"])
    final_df["tfidf_cosine"] = cosine_similarity(tfidf_job, tfidf_resume).diagonal()
    
    print("Computing Jaccard similarity...")
    final_df["jaccard"] = [
        jaccard_similarity(job, resume) 
        for job, resume in zip(final_df["job_text_clean"], final_df["resume_text_clean"])
    ]
    
    print("Computing BERT similarity...")
    try:
        bert_model = SentenceTransformer('all-mpnet-base-v2')
        job_embeddings = bert_model.encode(final_df["job_text_clean"], convert_to_tensor=True)
        resume_embeddings = bert_model.encode(final_df["resume_text_clean"], convert_to_tensor=True)
        final_df["bert_similarity"] = torch.nn.functional.cosine_similarity(job_embeddings, resume_embeddings).cpu().numpy()
    except Exception as e:
        print(f"Warning: BERT similarity calculation failed: {str(e)}")
        final_df["bert_similarity"] = 0
    
    print("Computing n-gram overlap...")
    final_df["ngram_overlap"] = [
        ngram_overlap(job, resume) 
        for job, resume in zip(final_df["job_text_clean"], final_df["resume_text_clean"])
    ]
    
    # Save processed data
    print(f"Saving processed data to {FEATURES_PATH}")
    final_df.to_csv(FEATURES_PATH, index=False)
    print("Text processing completed successfully!")
    
    return final_df

# Model Training Pipeline

def train_models(df=None):
    """Train ML models on the processed data."""
    print("Starting model training pipeline...")
    
    # Load processed data if not provided
    if df is None:
        try:
            print(f"Loading processed data from {FEATURES_PATH}")
            df = pd.read_csv(FEATURES_PATH)
        except FileNotFoundError:
            print(f"Error: Processed data file not found at {FEATURES_PATH}")
            print("Run the text processing pipeline first.")
            sys.exit(1)
    
    # Feature Scaling
    print("Scaling features...")
    scaler_features = MinMaxScaler()
    feature_columns = ["tfidf_cosine", "jaccard", "bert_similarity", "ngram_overlap"]
    X_scaled = scaler_features.fit_transform(df[feature_columns])
    
    # Train GMM
    print("Training GMM model...")
    num_components = 5
    gmm = GaussianMixture(n_components=num_components, random_state=42)
    gmm.fit(X_scaled)
    
    # Compute GMM Scores
    probabilities = gmm.predict_proba(X_scaled)
    gmm_scores = np.dot(probabilities, np.linspace(0, 100, num_components))
    
    # Normalize GMM Scores
    scaler_gmm = MinMaxScaler(feature_range=(0, 100))
    gmm_scores_scaled = scaler_gmm.fit_transform(gmm_scores.reshape(-1, 1)).flatten()
    df["gmm_match_score"] = gmm_scores_scaled
    
    # Plot Score Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(gmm_scores_scaled, bins=20, edgecolor="black", alpha=0.5)
    plt.title("GMM Match Score Distribution")
    plt.xlabel("Match Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(DATA_DIR, "outputs/gmm_score_distribution.png"))
    print(f"Score distribution plot saved to {os.path.join(DATA_DIR, 'outputs/gmm_score_distribution.png')}")
    
    # Save GMM & Scalers
    print(f"Saving models and scalers to {MODEL_DIR}")
    with open(os.path.join(MODEL_DIR, "gmm_model.pkl"), "wb") as f:
        pickle.dump(gmm, f)
    
    with open(os.path.join(MODEL_DIR, "scaler_gmm.pkl"), "wb") as f:
        pickle.dump(scaler_gmm, f)
    
    with open(os.path.join(MODEL_DIR, "scaler_features.pkl"), "wb") as f:
        pickle.dump(scaler_features, f)
    
    # Train ML Models
    print("Training ML models...")
    X = X_scaled
    y = df["gmm_match_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MAE": mae, "R² Score": r2}
        
        model_filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        with open(os.path.join(MODEL_DIR, model_filename), "wb") as f:
            pickle.dump(model, f)
        print(f"Saved {name} model to {os.path.join(MODEL_DIR, model_filename)}")
    
    # Print Evaluation Results
    print("\nModel Performance on GMM-Based Match Scores:")
    for model, metrics in results.items():
        print(f"{model}: MAE = {metrics['MAE']:.3f}, R² = {metrics['R² Score']:.3f}")
    
    # Save updated DataFrame with GMM score
    print(f"Saving final dataset to {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Model training completed successfully!")
    
    return df

# Main Function

def main():
    process_data()
    train_models()

if __name__ == "__main__":
    main() 