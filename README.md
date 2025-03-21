# **LlaMa-Resume-Optimizer**  
### **Try it live:** [🚀 LLaMA Resume Optimizer](https://llama-resume-optimizer.streamlit.app)  

### **Overview**  
This project is designed for **students and job seekers** who need to **instantly optimize their resumes** based on a specific **job description** to **pass ATS screenings** and improve recruiter evaluations.  

It uses **AI-driven analysis and optimization** to enhance resumes by:  
- **Matching keywords** from job descriptions.  
- **Improving readability and structure.**  
- **Providing ATS-friendly suggestions** using **machine learning and deep learning models**.  

By leveraging **LLaMA 3 fine-tuned with LoRA**, the system generates **context-aware edits** that ensure resumes align with job requirements, increasing the chances of getting noticed by recruiters. 🚀

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

We fine-tuned **LLaMA 3 (3B Instruct)** using **LoRA** on a custom dataset and uploaded it to Hugging Face as **["aaditey932/llama-resume-optimizer"](https://huggingface.co/aaditey932)**.  

**Dataset Creation:**  
- Machine learning **resumes** were sourced from **[Hire a Perfect Machine Learning Engineer](https://www.kaggle.com/datasets/sauravsolanki/hire-a-perfect-machine-learning-engineer)**.  
- AI-related **job descriptions** were sourced from **[Indeed Jobs Dataset](https://www.kaggle.com/datasets/vaghefi/indeed-jobs)**.  
- These were processed through **GPT-4o** to generate structured **resume improvement suggestions** in **JSON format**, like:  

```json
{
  "edit1": {
    "to_be_edited": "Led a team of software engineers to develop a recommendation engine for e-commerce.",
    "edited": "Managed a cross-functional team of 5 engineers to build a recommendation engine, increasing conversion rates by 20%.",
    "suggestions": "Consider adding a metric for user engagement improvement."
  }
}
```

**Workflow:**  
1. The **combined dataset (resume, job description, and AI-generated edits)** was used to fine-tune **LLaMA 3** via **LoRA**.  
2. The trained model suggests **ATS-friendly improvements** for **new resumes**.  
3. The system **automatically applies edits** by replacing `"to_be_edited"` with `"edited"` in the document.  

**Evaluation Metrics:**  
- ✅ **Keyword Inclusion Score** – Measures job-related keyword alignment.  
- ✅ **Semantic Similarity Score** – Assesses content improvement.  

This enables **context-aware** and **ATS-optimized** resume edits tailored to specific job descriptions. 🚀

---

## **User Interface**  
### **🔹 Streamlit Web App**  
- **Upload your resume** (`.pdf`, `.docx`) and enter a job description.  
- View **resume match score** and suggested **improvements**.  
- Download the **optimized ATS-friendly resume**.

🔗 **Try it live:** [LLaMA Resume Optimizer](https://llama-resume-optimizer.streamlit.app)  

---

## **Setup**  
Run the setup script to install dependencies and prepare the environment:  
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
Then, run the app:  
```bash
streamlit run app.py
```

---

## **Data Source**  
This project uses publicly available datasets to enhance resume optimization:

1. **[Hire a Perfect Machine Learning Engineer](https://www.kaggle.com/datasets/sauravsolanki/hire-a-perfect-machine-learning-engineer)** – Helps recruiters score resumes based on job descriptions using machine learning while guiding job seekers in resume optimization.

2. **[Indeed Jobs Dataset](https://www.kaggle.com/datasets/vaghefi/indeed-jobs)** – Provides job postings scraped from Indeed, including job descriptions, salaries, locations, and company details to help job seekers understand industry requirements and expectations.

---

## **Ethics Statement**  

This project uses publicly available datasets in compliance with their terms of use. We ensure that all data is handled responsibly, avoiding any misuse, unauthorized distribution, or unethical applications. No personally identifiable information (PII) is collected or used, and we strive to mitigate bias in AI-driven resume analysis. Our goal is to enhance fair and transparent hiring processes while respecting data privacy and ethical AI principles.

---

This version adds the **Streamlit app link at the start and under the UI section** while keeping everything structured and clean. 🚀 Let me know if you need further refinements!
