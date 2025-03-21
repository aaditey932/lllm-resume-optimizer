# **Resume Optimization**  
### **Overview**  
This project implements an **AI-powered resume analysis and optimization tool** that helps job seekers improve their resumes for **Applicant Tracking Systems (ATS)** and recruiter evaluations. It provides **keyword matching, readability enhancements, and recommendations** to increase hiring success rates.

---

## **Approaches**  
### **1️⃣ Naïve Approach**  
- Uses **keyword-based matching** to compare resumes with job descriptions keywords.  
- Calculates **overlap percentage** between resume text and job-related keywords.  
- Displays **top 3 job matches** with their match percentage.  

### **2️⃣ Machine Learning (ML) Approach**  
- **Text Preprocessing**: Cleaned job descriptions and resumes using lowercasing, punctuation removal, stopword filtering, and lemmatization.
- **Feature Engineering**: Computed four similarity metrics:
  - **TF-IDF Cosine Similarity**: Lexical overlap
  - **Jaccard Similarity**: Unique word overlap
  - **BERT Cosine Similarity**: Semantic similarity via embeddings
  - **N-Gram Overlap**: Phrase-level similarity (1–3 grams)
- **Match Scoring**: Applied a **Gaussian Mixture Model** to combine features into a composite match score (0–1 scale).
- **Modeling**: Trained multiple regression models to predict match quality from similarity features.
- **Analysis**: Used histograms and correlation heatmaps to interpret feature behavior.

### **3️⃣ Deep Learning Approach**  
This approach fine-tunes the **LLaMA 3 (3B Instruct)** model using **LoRA** on the combined dataset of resumes and job descriptions. This provides **ATS-friendly resume edits** in structured JSON format—suggesting improvements at the word, phrase, and sentence level.

---

## **User Interface**  
### **🔹 Streamlit Web App**  
- **Upload your resume** (`.pdf`, `.docx`) and enter a job description.  
- View **resume match score** and suggested **improvements**.  
- **Update**

---

## **Setup**  
Run the setup script to install dependencies and prepare the environment:  
- **Update**

---

## **Data Source**  
This project uses publicly available datasets to enhance resume optimization:

1. Hire a Perfect Machine Learning Engineer (https://www.kaggle.com/datasets/sauravsolanki/hire-a-perfect-machine-learning-engineer) – Helps recruiters score resumes based on job descriptions using machine learning while guiding job seekers in resume optimization.

2. Indeed Jobs Dataset (https://www.kaggle.com/datasets/vaghefi/indeed-jobs) – Provides job postings scraped from Indeed, including job descriptions, salaries, locations, and company details to help job seekers understand industry requirements and expectations.

---

## **Ethics Statement**  

This project uses publicly available datasets in compliance with their terms of use. We ensure that all data is handled responsibly, avoiding any misuse, unauthorized distribution, or unethical applications. No personally identifiable information (PII) is collected or used, and we strive to mitigate bias in AI-driven resume analysis. Our goal is to enhance fair and transparent hiring processes while respecting data privacy and ethical AI principles.

