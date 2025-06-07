import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
from docx import Document
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Job Recommendation Engine",
    page_icon="🤖",
    layout="wide",
)

# --- Configuration ---
MODEL_NAME = "adinplb/job-recommendation-sbert"
JOBS_DATA_URL = "https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Filtered_Jobs_4000.csv"
MAX_CV_UPLOADS = 10

# --- Helper Functions & Caching ---

@st.cache_resource
def load_model(model_name):
    """Loads a SentenceTransformer model from the Hugging Face Hub."""
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading the model '{model_name}': {e}")
        return None

@st.cache_data
def load_and_process_job_data(url):
    """Loads and preprocesses job data from a URL."""
    try:
        jobs_df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Failed to load data from URL: {e}")
        return None, []

    columns_to_combine = [
        "Title", "Position", "Company", "City", "State.Name", "Industry",
        "Job.Description", "Requirements", "Employment.Type", "Education.Required"
    ]
    existing_columns = [col for col in columns_to_combine if col in jobs_df.columns]
    
    jobs_df_filled = jobs_df[existing_columns].fillna("").astype(str)
    processed_texts = jobs_df_filled.agg(" ".join, axis=1).tolist()
    cleaned_texts = [text.replace("\n", " ").replace("\r", " ") for text in processed_texts]
    
    return jobs_df.copy(), cleaned_texts

@st.cache_data
def encode_corpus(_model, job_corpus_texts):
    """Encodes the job corpus using the provided model."""
    if not job_corpus_texts or _model is None:
        return None
    with st.spinner("Encoding the job database... This may take a moment on the first run."):
        corpus_embeddings = _model.encode(
            job_corpus_texts, convert_to_tensor=True, show_progress_bar=False
        )
    return corpus_embeddings

def parse_cv(uploaded_file):
    """Parses the text from an uploaded CV file (PDF or DOCX)."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    text = ""
    try:
        if file_extension == "pdf":
            # Read PDF content from in-memory buffer
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in pdf_document:
                text += page.get_text()
            pdf_document.close()
        elif file_extension == "docx":
            # Read DOCX content from in-memory buffer
            doc = Document(io.BytesIO(uploaded_file.read()))
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error parsing file {uploaded_file.name}: {e}")
        return ""
    return text

def display_recommendations(results, jobs_df):
    """Displays job recommendations in an expander format."""
    for i, (score, idx) in enumerate(zip(results.values, results.indices)):
        job_index = idx.item()
        original_job = jobs_df.iloc[job_index]
        
        expander_title = f"**{i+1}. {original_job.get('Title', 'N/A')}** at **{original_job.get('Company', 'N/A')}** (Score: {score.item():.2f})"
        
        with st.expander(expander_title):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Position:** {original_job.get('Position', 'N/A')}")
                st.markdown(f"**Location:** {original_job.get('City', 'N/A')}, {original_job.get('State.Name', 'N/A')}")
                st.markdown(f"**Employment Type:** {original_job.get('Employment.Type', 'N/A')}")
            with col2:
                st.markdown(f"**Industry:** {original_job.get('Industry', 'N/A')}")
                st.markdown(f"**Education Required:** {original_job.get('Education.Required', 'N/A')}")

            st.markdown("---")
            st.markdown("**Job Description:**")
            st.info(original_job.get('Job.Description', 'No description available.'))
            st.markdown("**Requirements:**")
            st.info(original_job.get('Requirements', 'No requirements listed.'))

# --- Main Application ---
st.title("📄 Job Recommendation Engine")

# Load model and data once
model = load_model(MODEL_NAME)
if model:
    jobs_df, job_corpus = load_and_process_job_data(JOBS_DATA_URL)
    corpus_embeddings = encode_corpus(model, job_corpus)

    # Create tabs for different search methods
    tab1, tab2 = st.tabs(["🔍 Search by Text Query", "📄 Search by CV Upload"])

    # --- Tab 1: Search by Text Query ---
    with tab1:
        st.header("Find Jobs Based on Your Query")
        st.markdown("Enter your resume summary, skills, or a description of your ideal job below.")
        
        user_query = st.text_area(
            "Enter your query here:",
            "Seeking a senior software engineer role specializing in backend development with Python, Django, and cloud services like AWS.",
            height=150, key="text_query"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            top_n_text = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10, key="slider_text")
        with col2:
            st.write("") # Spacer
            st.write("") # Spacer
            find_button_text = st.button("✨ Find Jobs", use_container_width=True, key="find_text")
        
        if find_button_text and user_query.strip():
            with st.spinner("Searching for the best matches..."):
                query_embedding = model.encode(user_query, convert_to_tensor=True)
                cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                top_results = torch.topk(cosine_scores, k=min(top_n_text, len(job_corpus)))
                st.success(f"Here are your top {top_n_text} recommendations based on your query:")
                display_recommendations(top_results, jobs_df)

    # --- Tab 2: Search by CV Upload ---
    with tab2:
        st.header("Find Jobs Based on Your CV")
        st.markdown(f"Upload up to **{MAX_CV_UPLOADS}** CVs in **PDF** or **DOCX** format.")
        
        uploaded_cvs = st.file_uploader(
            "Upload your CV(s) here",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="cv_uploader"
        )
        
        col1_cv, col2_cv = st.columns([3, 1])
        with col1_cv:
            top_n_cv = st.slider("Number of recommendations per CV:", min_value=5, max_value=20, value=5, key="slider_cv")
        
        if uploaded_cvs:
            if len(uploaded_cvs) > MAX_CV_UPLOADS:
                st.error(f"You can upload a maximum of {MAX_CV_UPLOADS} files. Please remove some files.")
            else:
                for cv_file in uploaded_cvs:
                    st.markdown(f"---")
                    st.subheader(f"Recommendations for: `{cv_file.name}`")
                    with st.spinner(f"Processing and finding matches for {cv_file.name}..."):
                        cv_text = parse_cv(cv_file)
                        if cv_text:
                            cv_embedding = model.encode(cv_text, convert_to_tensor=True)
                            cosine_scores = util.cos_sim(cv_embedding, corpus_embeddings)[0]
                            top_results = torch.topk(cosine_scores, k=min(top_n_cv, len(job_corpus)))
                            display_recommendations(top_results, jobs_df)
                        else:
                             st.warning(f"Could not extract text from {cv_file.name}. Skipping.")

else:
    st.error("The recommendation engine is currently unavailable. The model could not be loaded.")
