import kagglehub
import pandas as pd
import re
import pickle
import os
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# 📌 Define model directory
model_dir = "./lllm-resume-optimizer/models"
os.makedirs(model_dir, exist_ok=True)

# 📌 Step 1: Download Dataset from Kaggle
dataset_path = kagglehub.dataset_download("vaghefi/indeed-jobs")
file_path = dataset_path + "/indeed_jobs.csv"  # Adjust if filename is different

# 📌 Step 2: Load & Preprocess Data
df = pd.read_csv(file_path)
df = df.dropna(subset=['title', 'description'])
df['job_text'] = df['title'] + " " + df['description']

# 📌 Step 3: Clean Text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return words

df['Cleaned_Job_Text'] = df['job_text'].apply(clean_text)

# 📌 Step 4: Extract Keywords
category_keywords = {}
for title in df['title'].unique():
    all_words = [word for job in df[df['title'] == title]['Cleaned_Job_Text'] for word in job]
    most_common_words = Counter(all_words).most_common(20)
    category_keywords[title] = [word for word, freq in most_common_words]

# 📌 Step 5: Save to ./lllm-resume-optimizer/models
keyword_file_path = os.path.join(model_dir, "category_keywords.pkl")
with open(keyword_file_path, "wb") as file:
    pickle.dump(category_keywords, file)

print(f"✅ Keywords saved to: {keyword_file_path}")

## Above code generated using the DeepSeek & Chatgpt and then tweaked. 
