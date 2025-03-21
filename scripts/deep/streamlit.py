import scripts.deep.streamlit as st
import tempfile
import os
from scripts.deep.replace_text import replace_text_in_document
from scripts.deep.suggestions_gen import generate_suggestions
from PyPDF2 import PdfReader

st.title("📄 Resume Optimizer")
st.markdown("Upload your DOCX resume and enter a job description to get an optimized version with ATS-friendly edits.")

uploaded_file = st.file_uploader("Upload your .docx resume", type=["docx"])
job_description = st.text_area("Paste the job description here")

if uploaded_file is not None and job_description.strip():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name

    output_path = input_path.replace(".docx", "_edited.docx")

    # Extract resume text using PyPDF2 or any preferred method
    def extract_text_from_docx(docx_path):
        from docx import Document
        doc = Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])

    if st.button("✏️ Optimize Resume"):
        resume_text = extract_text_from_docx(input_path)
        with st.spinner("🔍 Generating suggestions..."):
            edits = generate_suggestions(resume_text, job_description)

        if edits:
            replace_text_in_document(input_path, output_path, edits)

            with open(output_path, "rb") as f:
                st.download_button(
                    label="📥 Download Optimized Resume",
                    data=f,
                    file_name="optimized_resume.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

            os.remove(output_path)
            os.remove(input_path)
        else:
            st.error("⚠️ Could not generate suggestions. Please try again.")
else:
    st.info("📎 Please upload a DOCX file and provide a job description to proceed.")
