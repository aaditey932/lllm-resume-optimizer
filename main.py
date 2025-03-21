import streamlit as st
import scripts.deep.gpt_resume_optimizer as st
import pandas as pd
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
import PyPDF2
import docx

# Download necessary NLTK data
nltk.download('punkt')

# Load Pretrained Job Keywords
@st.cache_resource
def load_keywords():
    with open("./models/category_keywords.pkl", "rb") as file:
        category_keywords = pickle.load(file)
    return category_keywords

category_keywords = load_keywords()
job_titles = list(category_keywords.keys())

# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    words = word_tokenize(text)  # Tokenize text
    return words  # Return cleaned word list

# Function to Extract Text from Uploaded Resume
def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

# Function to Match Resume Against Job Titles Using Keyword Overlap
def match_resume(resume_text):
    resume_words = clean_text(resume_text)

    match_scores = []
    for job_title, keywords in category_keywords.items():
        common_words = set(resume_words) & set(keywords)  # Find overlapping words
        match_score = len(common_words) / len(keywords) if keywords else 0  # Calculate match percentage
        match_scores.append((job_title, match_score))

    # Sort and get top 3 job matches
    match_results = sorted(match_scores, key=lambda x: x[1], reverse=True)[:3]
    return match_results

# Streamlit UI
st.title("📄 AI-Powered Resume Optimization System")
st.write("Upload your resume for recommendations")

# File Upload
uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    resume_text = extract_text_from_file(uploaded_file)
    
    if resume_text:
        st.subheader("Naive Approach: Keyword Matching - Top 3 Job Matches:")
        match_results = match_resume(resume_text)
        
        for job_title, score in match_results:
            st.write(f"✅ **{job_title}: {score*100:.2f}% match**")
    else:
        st.error("Error extracting text from the uploaded file. Please try another format.")


## Above code generated using the DeepSeek & Chatgpt and then tweaked. 
