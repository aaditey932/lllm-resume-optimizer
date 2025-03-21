import os
import json
import spacy
import yake
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from docx import Document  # For DOCX text extraction

# Load NLP tools
nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_keywords(text, top_n=10):
    """Extract keywords using YAKE and spaCy."""
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=top_n)
    yake_keywords = {kw for kw, _ in kw_extractor.extract_keywords(text)}

    doc = nlp(text)
    spacy_keywords = {token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"]}

    all_keywords = list(yake_keywords | spacy_keywords)
    return sorted(all_keywords, key=lambda x: text.count(x), reverse=True)[:top_n]

def keyword_inclusion_score(job_description, resume_text):
    """Calculates the percentage of job description keywords included in the resume."""
    keywords = extract_keywords(job_description)
    included = [keyword for keyword in keywords if keyword in resume_text]
    return len(included) / len(keywords) if keywords else 0

def semantic_similarity(original_text, edited_text):
    """Calculates semantic similarity between two texts."""
    embeddings = sentence_model.encode([original_text, edited_text])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def evaluate_resume(original_resume_path, optimized_resume_path, job_description):
    """Evaluates the optimized resume against the original resume and job description."""
    # Extract text
    if original_resume_path.endswith(".pdf"):
        original_text = extract_text_from_pdf(original_resume_path)
    else:
        original_text = extract_text_from_docx(original_resume_path)

    if optimized_resume_path.endswith(".pdf"):
        optimized_text = extract_text_from_pdf(optimized_resume_path)
    else:
        optimized_text = extract_text_from_docx(optimized_resume_path)

    # Compute scores
    keyword_score = keyword_inclusion_score(job_description, optimized_text)
    similarity = semantic_similarity(original_text, optimized_text)

    return {
        "keyword_inclusion_score": round(keyword_score, 2),
        "semantic_similarity": round(similarity, 2)
    }

# Example usage
if __name__ == "__main__":
    original_resume_path = "/data/raw/Sample_Aaditey_Pillai_Resume.docx"
    optimized_resume_path = "/data/outputs/optimized_resume.docx"

    job_description = """
    About the job
    Lab Summary: AI Research Center (AIC) located in Mountain View, California focuses on research and development which directly impacts future Samsung products reaching hundreds of millions of users worldwide...

    Required Skills 

    - Teamwork and communication skills
    - Current Ph.D. student in CS, EE, or related field
    - Experience in machine learning (e.g., supervised, transfer, few-shot learning)
    - Expertise in LLMs, RAG, reasoning, action planning
    - Proficiency in PyTorch or TensorFlow
    - Publications in NeurIPS, ICLR, ACL, etc.
    """

    results = evaluate_resume(original_resume_path, optimized_resume_path, job_description)
    print(json.dumps(results, indent=2))
