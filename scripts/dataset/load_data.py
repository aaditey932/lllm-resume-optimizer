#!/usr/bin/env python3
"""
Load Data Script
---------------
This script downloads the required datasets from Kaggle and loads them into the
appropriate project folders, including train/test split for resumes.
"""

import os
import kagglehub
import pandas as pd
from tqdm import tqdm

def main():
    """Main function to execute data downloading and loading"""
    # Download datasets from kaggle
    print("Downloading datasets from Kaggle...")
    path_jd = kagglehub.dataset_download("vaghefi/indeed-jobs")
    path_resumes = kagglehub.dataset_download("sauravsolanki/hire-a-perfect-machine-learning-engineer")

    print(f"Job Descriptions Path: {path_jd}")
    print(f"Resumes Path: {path_resumes}")

    # Define project paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
    RAW_JOBS_PATH = os.path.join(PROJECT_ROOT, "data/raw/jobs")

    # Ensure directories exist
    os.makedirs(RAW_JOBS_PATH, exist_ok=True)

    # Load job descriptions CSV
    job_descriptions = pd.read_csv(os.path.join(path_jd, "indeed_jobs.csv"))

    # Save the CSV file in the correct directory
    job_descriptions.to_csv(os.path.join(RAW_JOBS_PATH, "job_descriptions.csv"), index=False)

    print(f"Job descriptions CSV saved at: {os.path.join(RAW_JOBS_PATH, 'job_descriptions.csv')}")
    print(f"Loaded {len(job_descriptions)} job descriptions")

    # Define paths for training resumes
    RAW_RESUMES_PATH = os.path.join(PROJECT_ROOT, "data/raw/resumes/train")
    os.makedirs(RAW_RESUMES_PATH, exist_ok=True)

    # Source folder where kagglehub stores training pdfs
    pdf_folder = os.path.join(path_resumes, "HireAMLE/dataset/trainResumes")
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    # Save training pdfs
    for pdf_file in tqdm(pdf_files, desc="Downloading PDFs to /data/raw/resumes/train"):
        source_path = os.path.join(pdf_folder, pdf_file)
        destination_path = os.path.join(RAW_RESUMES_PATH, pdf_file)
        
        if not os.path.exists(destination_path):  # Avoid redundant downloads
            with open(source_path, "rb") as src_file, open(destination_path, "wb") as dest_file:
                dest_file.write(src_file.read())

    print(f"All training resumes (PDFs) have been downloaded and saved to: {RAW_RESUMES_PATH}")
    print(f"Total Training Resumes Downloaded: {len(pdf_files)}")

    # Download the test resumes
    RAW_TEST_RESUMES_PATH = os.path.join(PROJECT_ROOT, "data/raw/resumes/test")
    os.makedirs(RAW_TEST_RESUMES_PATH, exist_ok=True)

    # Source folder where kagglehub stores test pdfs
    test_resumes_path = os.path.join(path_resumes, "HireAMLE/dataset/testResumes")
    test_pdf_files = [f for f in os.listdir(test_resumes_path) if f.endswith(".pdf")]

    # Save test resumes
    for pdf_file in tqdm(test_pdf_files, desc="Downloading test resumes to /data/raw/resumes/test"):
        source_path = os.path.join(test_resumes_path, pdf_file)
        destination_path = os.path.join(RAW_TEST_RESUMES_PATH, pdf_file)

        if not os.path.exists(destination_path):  # Avoid redundant downloads
            with open(source_path, "rb") as src_file, open(destination_path, "wb") as dest_file:
                dest_file.write(src_file.read())

    print(f"All test resumes (PDFs) have been downloaded and saved to: {RAW_TEST_RESUMES_PATH}")
    print(f"Total Test Resumes Downloaded: {len(test_pdf_files)}")
    
    print("Data download and loading completed successfully!")
    return {
        "project_root": PROJECT_ROOT,
        "raw_jobs_path": RAW_JOBS_PATH,
        "raw_train_resumes_path": RAW_RESUMES_PATH,
        "raw_test_resumes_path": RAW_TEST_RESUMES_PATH,
        "job_descriptions_count": len(job_descriptions),
        "train_resumes_count": len(pdf_files),
        "test_resumes_count": len(test_pdf_files)
    }

if __name__ == "__main__":
    main()


# This script was created using ChatGPT-4o and cursor. The code was first written and finalized in a notebook format. 
# Then ChatGPT was asked to convert the code into a functional and working script. 