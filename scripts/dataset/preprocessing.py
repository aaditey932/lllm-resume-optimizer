#!/usr/bin/env python3
"""
Preprocessing Script
------------------
This script processes the downloaded data, extracts text from PDFs,
and creates the final dataset without additional preprocessing.
"""

import os
import pandas as pd
from tqdm import tqdm
from PyPDF2 import PdfReader

def get_project_root():
    """Get the project root directory path"""
    # Assuming this script is in scripts/dataset/ folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get to project root
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    return project_root

def setup_directories(project_root):
    """Create all necessary directories if they don't exist"""
    directories = [
        os.path.join(project_root, "data/processed/jobs"),
        os.path.join(project_root, "data/processed/resumes"),
        os.path.join(project_root, "data/outputs")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return {
        "raw_jobs_path": os.path.join(project_root, "data/raw/jobs"),
        "raw_train_resumes_path": os.path.join(project_root, "data/raw/resumes/train"),
        "raw_test_resumes_path": os.path.join(project_root, "data/raw/resumes/test"),
        "processed_jobs_path": directories[0],
        "processed_resumes_path": directories[1],
        "outputs_path": directories[2]
    }

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def process_resumes(paths_dict):
    """Process resume PDFs and extract text"""
    print("Processing training resumes...")
    
    # Get list of training PDFs
    train_pdf_files = [f for f in os.listdir(paths_dict["raw_train_resumes_path"]) if f.endswith(".pdf")]
    
    # Extract text from each training PDF and store it in a DataFrame
    train_resume_data = []
    for pdf_file in tqdm(train_pdf_files, desc="Extracting text from training PDFs"):
        pdf_path = os.path.join(paths_dict["raw_train_resumes_path"], pdf_file)
        resume_text = extract_text_from_pdf(pdf_path)
        train_resume_data.append({
            "filename": pdf_file, 
            "resume_text": resume_text,
            "set": "train"
        })
    
    # Convert to DataFrame
    df_train_resumes = pd.DataFrame(train_resume_data)
    
    # Save processed training resumes
    train_output_path = os.path.join(paths_dict["processed_resumes_path"], "processed_train_resumes.csv")
    df_train_resumes.to_csv(train_output_path, index=False)
    
    print(f"Processed {len(df_train_resumes)} training resumes and saved to {train_output_path}")
    
    # Process test resumes
    print("Processing test resumes...")
    
    # Get list of test PDFs
    test_pdf_files = [f for f in os.listdir(paths_dict["raw_test_resumes_path"]) if f.endswith(".pdf")]
    
    # Extract text from each test PDF and store it in a DataFrame
    test_resume_data = []
    for pdf_file in tqdm(test_pdf_files, desc="Extracting text from test PDFs"):
        pdf_path = os.path.join(paths_dict["raw_test_resumes_path"], pdf_file)
        resume_text = extract_text_from_pdf(pdf_path)
        test_resume_data.append({
            "filename": pdf_file, 
            "resume_text": resume_text,
            "set": "test"
        })
    
    # Convert to DataFrame
    df_test_resumes = pd.DataFrame(test_resume_data)
    
    # Save processed test resumes
    test_output_path = os.path.join(paths_dict["processed_resumes_path"], "processed_test_resumes.csv")
    df_test_resumes.to_csv(test_output_path, index=False)
    
    print(f"Processed {len(df_test_resumes)} test resumes and saved to {test_output_path}")
    
    # Combine train and test resumes
    df_all_resumes = pd.concat([df_train_resumes, df_test_resumes], ignore_index=True)
    
    # Save combined resumes
    combined_output_path = os.path.join(paths_dict["processed_resumes_path"], "processed_all_resumes.csv")
    df_all_resumes.to_csv(combined_output_path, index=False)
    
    print(f"Combined {len(df_all_resumes)} total resumes and saved to {combined_output_path}")
    
    return df_train_resumes, df_test_resumes, df_all_resumes

def process_job_descriptions(paths_dict):
    """Process job descriptions data"""
    print("Processing job descriptions...")
    
    # Load job descriptions
    job_descriptions = pd.read_csv(os.path.join(paths_dict["raw_jobs_path"], "job_descriptions.csv"))
    
    # Format job descriptions
    job_descriptions["job_description"] = "**TITLE**: " + job_descriptions["title"] + \
                                         " **DESCRIPTIONS** " + job_descriptions["description"] + \
                                         " **COMPANY** " + job_descriptions["company"]
    
    # Drop all columns except job_description
    job_descriptions = job_descriptions[["job_description"]]
    
    # Save processed job descriptions
    output_path = os.path.join(paths_dict["processed_jobs_path"], "processed_job_descriptions.csv")
    job_descriptions.to_csv(output_path, index=False)
    
    print(f"Processed {len(job_descriptions)} job descriptions and saved to {output_path}")
    return job_descriptions

def create_final_dataset(job_descriptions, df_all_resumes, paths_dict, sample_size=2000):
    """Create the final dataset by matching job descriptions with resumes"""
    print("Creating final dataset...")
    
    # Sample job descriptions (if needed)
    if len(job_descriptions) > sample_size:
        job_descriptions_sampled = job_descriptions.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        job_descriptions_sampled = job_descriptions.copy().reset_index(drop=True)
    
    # Handle case where we have fewer resumes than job descriptions
    num_resumes = len(df_all_resumes)
    print(f"Total resumes available: {num_resumes}")
    
    if num_resumes < sample_size:
        print(f"Only {num_resumes} resumes available, will repeat to match {sample_size} job descriptions")
        repeated_resumes = [df_all_resumes.iloc[i % num_resumes] for i in range(sample_size)]
        df_resumes_expanded = pd.DataFrame(repeated_resumes).reset_index(drop=True)
    else:
        df_resumes_expanded = df_all_resumes.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Combine job descriptions and resumes
    final_df = job_descriptions_sampled.copy()
    final_df["resume_text"] = df_resumes_expanded["resume_text"].values
    
    # Save final dataset
    final_path = os.path.join(paths_dict["outputs_path"], "matched_resumes_jobs.csv")
    final_df.to_csv(final_path, index=False)
    
    print(f"Final dataset saved at: {final_path}")
    print(f"Final dataset contains {len(final_df)} job-resume pairs")
    
    return final_df

def main():
    """Main function to execute data preprocessing"""
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    # Setup directory structure
    paths_dict = setup_directories(project_root)
    
    # Process resumes
    df_train_resumes, df_test_resumes, df_all_resumes = process_resumes(paths_dict)
    
    # Process job descriptions
    job_descriptions = process_job_descriptions(paths_dict)
    
    # Create final dataset using all resumes
    final_df = create_final_dataset(job_descriptions, df_all_resumes, paths_dict)
    
    print("Preprocessing completed successfully!")
    return {
        "train_resumes_count": len(df_train_resumes),
        "test_resumes_count": len(df_test_resumes),
        "all_resumes_count": len(df_all_resumes),
        "job_descriptions_count": len(job_descriptions),
        "final_dataset_count": len(final_df)
    }

if __name__ == "__main__":
    main()


# This script was created using ChatGPT-4o and cursor. The code was first written and finalized in a notebook format. 
# Then ChatGPT was asked to convert the code into a functional and working script. 