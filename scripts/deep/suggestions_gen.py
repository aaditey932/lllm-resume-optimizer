import kagglehub
import pandas as pd
import os
import json
from PyPDF2 import PdfReader
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
OPEN_AI_API = os.getenv("OPEN_AI_API")

# Initialize OpenAI client
client = OpenAI(api_key=OPEN_AI_API)

def download_datasets():
    """Download datasets from Kaggle Hub."""
    path_jd = kagglehub.dataset_download("vaghefi/indeed-jobs")
    path_resumes = kagglehub.dataset_download("sauravsolanki/hire-a-perfect-machine-learning-engineer")
    print(f"Job descriptions downloaded to: {path_jd}")
    print(f"Resumes downloaded to: {path_resumes}")
    return path_jd, path_resumes

def load_job_descriptions(path_jd):
    """Load and preprocess job descriptions."""
    job_descriptions = pd.read_csv(path_jd + "/indeed_jobs.csv")
    job_descriptions["job_description"] = (
        "TITLE: " + job_descriptions["title"] + " DESCRIPTIONS: " + job_descriptions["description"] + " COMPANY: " + job_descriptions["company"]
    )
    job_descriptions = job_descriptions.drop(
        columns=["title", "description", "company", "city", "state", "zipcode", "salary", "rating", "reviews"]
    )
    return job_descriptions

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def load_resumes(pdf_folder):
    """Load and preprocess resumes from PDF files."""
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    data = []
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        resume_text = extract_text_from_pdf(pdf_path)
        data.append({"resumes": resume_text})
    return pd.DataFrame(data)

def generate_suggestions(resume_text, job_description):
    """Generate structured resume optimization suggestions in JSON format."""
    prompt = f"""
You are an AI-powered resume optimization assistant. Your task is to analyze the provided resume against the job description and generate ATS-friendly, structured, and impact-driven improvements. These improvements should be directly replaceable in the original document.

### Key Requirements:
- Output will be used programmatically to modify the resume while maintaining its structure.
- Ensure maximum ATS compatibility by:
  - Embedding relevant keywords from the job description naturally.
  - Using active, impact-driven language to highlight achievements.
  - Making job titles, dates, and formatting consistent.
- Response must be a structured JSON object.

### Output Format:
Generate a structured JSON object with numbered edits (`"edit1"`, `"edit2"`, etc.) where:
- `"to_be_edited"` → The exact resume text that needs improvement.
- `"edited"` → The optimized version with clear, ATS-friendly, and impact-driven phrasing.
  - Include missing keywords from the job description.
  - Add impact-driven metrics (e.g., "Increased efficiency by 30%").
  - Use concise phrasing improvements (e.g., replace "responsible for" with "managed").
  - Fix formatting issues (e.g., consistent bullet points or date formats).

### Instructions:
1. **Word-level**: Replace weak, generic words with stronger ATS-friendly alternatives.
2. **Phrase-level**: Refine short phrases for clarity, conciseness, and keyword inclusion.
3. **Sentence-level**: Enhance structure to highlight quantifiable impact and results.
4. **Skills**: Ensure any skills mentioned in the job description are included in the edits.

---

### Resume:
{resume_text}

### Job Description:
{job_description}

---

### Expected JSON Output Example:
{{
    "edit1": {{
        "to_be_edited": "Led a team of software engineers to develop a recommendation engine for e-commerce.",
        "edited": "Managed a cross-functional team of 5 engineers to build a recommendation engine, increasing conversion rates by 20%.",
    }},
    "edit2": {{
        "to_be_edited": "Worked with Python, SQL, and machine learning models to optimize product recommendations.",
        "edited": "Developed and deployed machine learning models using Python and SQL, improving recommendation accuracy by 15%.",
    }},
    "edit3": {{
        "to_be_edited": "Developed various data science projects in Python and R.",
        "edited": "Designed and implemented multiple data science projects in Python and R, including predictive modeling and NLP applications.",
    }}
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        response_format={"type": "json_object"}
    )
    suggestions_json = response.choices[0].message.content
    try:
        return json.loads(suggestions_json)
    except (json.JSONDecodeError, TypeError):
        print("❌ Warning: API response is not a valid JSON object.")
        return None

def process_and_save():
    """Process job descriptions, resumes, generate suggestions, and save the results."""
    path_jd, path_resumes = download_datasets()

    # Load job descriptions
    job_descriptions = load_job_descriptions(path_jd)

    # Load resumes
    pdf_folder = path_resumes + "/HireAMLE/dataset/trainResumes"
    resumes_df = load_resumes(pdf_folder)

    # Sample job descriptions
    job_descriptions_sampled = job_descriptions.sample(n=3000, random_state=42).reset_index(drop=True)

    # Expand resumes to match job descriptions count
    num_resumes = len(resumes_df)
    repeated_resumes = [resumes_df.iloc[i % num_resumes] for i in range(3000)]
    df_resumes_expanded = pd.DataFrame(repeated_resumes).reset_index(drop=True)

    # Create final dataframe
    final_df = job_descriptions_sampled.copy()
    final_df["resume"] = df_resumes_expanded["resumes"].values  # Assign resumes
    final_df["suggestions"] = ""

    # Generate and store suggestions
    for i in tqdm(range(len(final_df)), desc="Generating Suggestions"):
        resume_text = final_df.loc[i, "resume"]
        job_description = final_df.loc[i, "job_description"]

        suggestions = generate_suggestions(resume_text, job_description)
        time.sleep(2)  # Avoid rate limits
        
        final_df.loc[i, "suggestions"] = json.dumps(suggestions)

    # Save to CSV
    save_path = "/data/processed/suggestions_json.csv"
    final_df.to_csv(save_path, index=False)
    print(f"✅ Suggestions saved to {save_path}")

# Run the process
if __name__ == "__main__":
    process_and_save()
