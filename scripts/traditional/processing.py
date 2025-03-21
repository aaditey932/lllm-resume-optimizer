import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import torch
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Set paths
PROJECT_ROOT = "/Users/owaiskamdar/Desktop/resume_optimizer/lllm-resume-optimizer"
FINAL_JOBS_PATH = os.path.join(PROJECT_ROOT, "data/outputs/matched_resumes_jobs.csv")
JOBS_WITH_FEATURES_PATH = os.path.join(PROJECT_ROOT, "data/outputs/matched_resumes_jobs_withfeatures.csv")
# Load dataset
final_df = pd.read_csv(FINAL_JOBS_PATH)

# Step 1: Text Preprocessing
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def remove_markup(text):
    text = re.sub(r'\*\*[^*]+\*\*', ' ', text)  # Remove markdown bold
    text = re.sub(r'#\S+', ' ', text)  # Remove hashtags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(token) for token in text.split() if token not in stop_words]
    return " ".join(tokens)

final_df["job_text_clean"] = final_df["job_description"].apply(lambda x: preprocess_text(remove_markup(str(x))))
final_df["resume_text_clean"] = final_df["resume_text"].apply(lambda x: preprocess_text(remove_markup(str(x))))

# Step 2: Compute Similarity Scores

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=15, max_df=0.9)
tfidf_job = tfidf_vectorizer.fit_transform(final_df["job_text_clean"])
tfidf_resume = tfidf_vectorizer.transform(final_df["resume_text_clean"])
final_df["tfidf_cosine"] = cosine_similarity(tfidf_job, tfidf_resume).diagonal()

# Jaccard Similarity
def jaccard_similarity(text1, text2):
    set1, set2 = set(text1.split()), set(text2.split())
    return len(set1 & set2) / len(set1 | set2) if set1 and set2 else 0

final_df["jaccard"] = [jaccard_similarity(job, resume) for job, resume in zip(final_df["job_text_clean"], final_df["resume_text_clean"])]

# BERT Similarity
bert_model = SentenceTransformer('all-mpnet-base-v2')
job_embeddings = bert_model.encode(final_df["job_text_clean"], convert_to_tensor=True)
resume_embeddings = bert_model.encode(final_df["resume_text_clean"], convert_to_tensor=True)
final_df["bert_similarity"] = torch.nn.functional.cosine_similarity(job_embeddings, resume_embeddings).cpu().numpy()

# N-Gram Overlap
def ngram_overlap(text1, text2, ngram_range=(1,3)):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    ngrams1 = set(vectorizer.fit([text1]).get_feature_names_out())
    ngrams2 = set(vectorizer.fit([text2]).get_feature_names_out())
    return len(ngrams1 & ngrams2) / max(len(ngrams1 | ngrams2), 1)

final_df["ngram_overlap"] = [ngram_overlap(job, resume) for job, resume in zip(final_df["job_text_clean"], final_df["resume_text_clean"])]

# Step 3: Save Processed Data
final_df.to_csv(JOBS_WITH_FEATURES_PATH, index=False)
print("Processed data saved successfully at:", JOBS_WITH_FEATURES_PATH)

