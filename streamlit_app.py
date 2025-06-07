import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util, models, losses, InputExample
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader
import fitz  # PyMuPDF
from docx import Document
import io
import os
import logging
import nltk 

nltk.download('punkt_tab')

# --- Page and Logging Configuration ---
st.set_page_config(
    page_title="Job Recommendation Engine",
    page_icon="🤖",
    layout="wide",
)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model & Data Configuration ---
# This will be the name of the folder where your final model is stored
FINAL_MODEL_PATH = "trained_job_recommender_model" 
# Base model for training, if needed
BASE_MODEL_NAME_FOR_TRAINING = 'sentence-transformers/all-MiniLM-L6-v2'
JOBS_DATA_URL = "https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Filtered_Jobs_4000.csv"
ONET_DATA_URL = "https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Occupation%20Data.csv"

# --- Training Hyperparameters ---
TSDAE_EPOCHS = 1
TSDAE_BATCH_SIZE = 8  # Reduced for Streamlit Cloud's memory limits
TSDAE_LEARNING_RATE = 3e-5
TSDAE_MAX_SEQ_LENGTH = 256

SBERT_EPOCHS = 1
SBERT_BATCH_SIZE = 8 # Reduced for Streamlit Cloud's memory limits
SBERT_LEARNING_RATE = 2e-5

# --- Helper Functions for Data Processing ---
def process_jobs_csv_for_tsdae(filepath_or_df):
    try:
        if isinstance(filepath_or_df, str):
            jobs_df = pd.read_csv(filepath_or_df)
        else:
            jobs_df = filepath_or_df.copy()
    except Exception as e:
        logger.error(f"Error processing jobs data source: {e}")
        return None, []

    columns_to_combine = [
        'Title', 'Position', 'Company', 'City', 'State.Name',
        'Industry', 'Job.Description', 'Requirements', 'Employment.Type', 'Education.Required'
    ]
    existing_columns = [col for col in columns_to_combine if col in jobs_df.columns]
    
    jobs_df_filled = jobs_df[existing_columns].fillna('').astype(str)
    processed_texts = jobs_df_filled.agg(' '.join, axis=1).tolist()
    cleaned_texts = [text.replace('\n', ' ').replace('\r', ' ') for text in processed_texts]
    return jobs_df.copy(), cleaned_texts

def process_onet_csv_for_sbert_training(filepath_or_df):
    examples = []
    try:
        if isinstance(filepath_or_df, str):
            onet_df = pd.read_csv(filepath_or_df)
        else:
            onet_df = filepath_or_df.copy()
    except Exception as e:
        logger.error(f"Error processing ONET data source: {e}")
        return []
        
    for _, row in onet_df.iterrows():
        title = str(row['Title']).replace('\n', ' ').replace('\r', ' ')
        description = str(row['Description']).replace('\n', ' ').replace('\r', ' ')
        examples.append(InputExample(texts=[title, description], label=1.0))
    return examples

# --- Full Training Pipeline ---
def train_model_pipeline(jobs_data_src, onet_data_src, base_model, final_save_path):
    st.info("Starting model training pipeline. This is a one-time process and may take several minutes.")
    
    # --- Stage 1: TSDAE Pre-training ---
    st.write("--- Stage 1: TSDAE Pre-training on Job Descriptions ---")
    progress_bar = st.progress(0, text="Processing job data for TSDAE...")
    
    _, train_sentences_tsdae = process_jobs_csv_for_tsdae(jobs_data_src)
    if not train_sentences_tsdae:
        st.error("TSDAE training failed: No job data processed.")
        return False

    progress_bar.progress(10, text="Defining TSDAE model...")
    word_embedding_model = models.Transformer(base_model, max_seq_length=TSDAE_MAX_SEQ_LENGTH)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    tsdae_train_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    tsdae_dataset = DenoisingAutoEncoderDataset(train_sentences_tsdae)
    tsdae_dataloader = DataLoader(tsdae_dataset, batch_size=TSDAE_BATCH_SIZE, shuffle=True)
    tsdae_loss = losses.DenoisingAutoEncoderLoss(model=tsdae_train_model, decoder_name_or_path=base_model, tie_encoder_decoder=True)

    progress_bar.progress(20, text="Training TSDAE model... (This is the longest step)")
    tsdae_train_model.fit(
        train_objectives=[(tsdae_dataloader, tsdae_loss)], epochs=TSDAE_EPOCHS,
        weight_decay=0, scheduler='WarmupLinear', optimizer_params={'lr': TSDAE_LEARNING_RATE},
        warmup_steps=100, show_progress_bar=False, use_amp=True
    )
    
    temp_tsdae_output_path = 'temp_tsdae_model'
    os.makedirs(temp_tsdae_output_path, exist_ok=True)
    tsdae_train_model.save(temp_tsdae_output_path)
    
    # --- Stage 2: SBERT Fine-tuning ---
    st.write("--- Stage 2: Fine-tuning on ONET dataset ---")
    progress_bar.progress(70, text="Processing ONET data for fine-tuning...")

    sbert_model_to_finetune = SentenceTransformer(temp_tsdae_output_path)
    sbert_train_samples = process_onet_csv_for_sbert_training(onet_data_src)
    if not sbert_train_samples:
        st.error("SBERT fine-tuning failed: No ONET data processed.")
        return False

    num_train_steps_sbert = len(sbert_train_samples) // SBERT_BATCH_SIZE * SBERT_EPOCHS
    sbert_warmup_steps = int(0.1 * num_train_steps_sbert) if num_train_steps_sbert > 0 else 0
    
    sbert_train_dataloader_mnrl = DataLoader(sbert_train_samples, shuffle=True, batch_size=SBERT_BATCH_SIZE)
    sbert_loss_mnrl = losses.MultipleNegativesRankingLoss(model=sbert_model_to_finetune)

    progress_bar.progress(80, text="Fine-tuning SBERT model...")
    sbert_model_to_finetune.fit(
        train_objectives=[(sbert_train_dataloader_mnrl, sbert_loss_mnrl)], epochs=SBERT_EPOCHS,
        warmup_steps=sbert_warmup_steps, optimizer_params={'lr': SBERT_LEARNING_RATE},
        weight_decay=0.01, show_progress_bar=False, use_amp=True, save_best_model=False
    )
    
    progress_bar.progress(95, text="Saving final model...")
    os.makedirs(final_save_path, exist_ok=True)
    sbert_model_to_finetune.save(final_save_path)
    
    progress_bar.progress(100, text="Model training complete!")
    st.success(f"Model successfully trained and saved to: {final_save_path}")
    return True

# --- Main Application Logic ---

@st.cache_resource
def get_model():
    """
    Checks for a trained model. If not found, it runs the training pipeline.
    If found, it loads the model.
    """
    if not os.path.exists(FINAL_MODEL_PATH):
        st.warning("Custom-trained model not found. Starting one-time training process.")
        training_successful = train_model_pipeline(
            JOBS_DATA_URL,
            ONET_DATA_URL,
            BASE_MODEL_NAME_FOR_TRAINING,
            FINAL_MODEL_PATH
        )
        if not training_successful:
            st.error("Model training failed. Please check the logs.")
            return None
    
    try:
        model = SentenceTransformer(FINAL_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the trained model from {FINAL_MODEL_PATH}: {e}")
        return None

# Rest of the app (caching, parsing, display) remains mostly the same
@st.cache_data
def load_job_data(url):
    try:
        jobs_df = pd.read_csv(url)
        return jobs_df
    except Exception as e:
        st.error(f"Failed to load data from URL: {e}")
        return None

@st.cache_data
def get_job_corpus(jobs_df):
    if jobs_df is None:
        return []
    _, cleaned_texts = process_jobs_csv_for_tsdae(jobs_df)
    return cleaned_texts

@st.cache_data
def encode_corpus(_model, job_corpus_texts):
    if not job_corpus_texts or _model is None:
        return None
    with st.spinner("Encoding the job database... This may take a moment."):
        corpus_embeddings = _model.encode(
            job_corpus_texts, convert_to_tensor=True, show_progress_bar=False
        )
    return corpus_embeddings

def parse_cv(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    text = ""
    try:
        if file_extension == "pdf":
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in pdf_document:
                text += page.get_text()
            pdf_document.close()
        elif file_extension == "docx":
            doc = Document(io.BytesIO(uploaded_file.read()))
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error parsing file {uploaded_file.name}: {e}")
        return ""
    return text

def display_recommendations(results, jobs_df):
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

# --- Main Application UI ---
st.title("📄 Advanced Job Recommendation Engine")

model = get_model()

if model:
    jobs_df = load_job_data(JOBS_DATA_URL)
    job_corpus = get_job_corpus(jobs_df)
    corpus_embeddings = encode_corpus(model, job_corpus)

    if corpus_embeddings is not None:
        st.success("Custom-trained model loaded successfully!")
        tab1, tab2 = st.tabs(["🔍 Search by Text Query", "📄 Search by CV Upload"])

        with tab1:
            # ... (UI code for text query remains the same)
            st.header("Find Jobs Based on Your Query")
            st.markdown("Enter your resume summary, skills, or a description of your ideal job below.")
            
            user_query = st.text_area(
                "Enter your query here:", "Seeking a senior software engineer role specializing in backend development with Python, Django, and cloud services like AWS.",
                height=150, key="text_query"
            )

            col1, col2 = st.columns([3, 1])
            with col1:
                top_n_text = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10, key="slider_text")
            with col2:
                st.write(""); st.write("") # Spacers
                find_button_text = st.button("✨ Find Jobs", use_container_width=True, key="find_text")
            
            if find_button_text and user_query.strip():
                with st.spinner("Searching for matches..."):
                    query_embedding = model.encode(user_query, convert_to_tensor=True)
                    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                    top_results = torch.topk(cosine_scores, k=min(top_n_text, len(job_corpus)))
                    st.success(f"Here are your top {top_n_text} recommendations:")
                    display_recommendations(top_results, jobs_df)

        with tab2:
            # ... (UI code for CV upload remains the same)
            st.header("Find Jobs Based on Your CV")
            st.markdown(f"Upload up to **10** CVs in **PDF** or **DOCX** format.")
            
            uploaded_cvs = st.file_uploader(
                "Upload your CV(s) here", type=["pdf", "docx"], accept_multiple_files=True, key="cv_uploader"
            )
            
            top_n_cv = st.slider("Number of recommendations per CV:", min_value=5, max_value=20, value=5, key="slider_cv")
            
            if uploaded_cvs:
                if len(uploaded_cvs) > 10:
                    st.error(f"You can upload a maximum of 10 files.")
                else:
                    for cv_file in uploaded_cvs:
                        st.markdown(f"---")
                        st.subheader(f"Recommendations for: `{cv_file.name}`")
                        with st.spinner(f"Processing {cv_file.name}..."):
                            cv_text = parse_cv(cv_file)
                            if cv_text:
                                cv_embedding = model.encode(cv_text, convert_to_tensor=True)
                                cosine_scores = util.cos_sim(cv_embedding, corpus_embeddings)[0]
                                top_results = torch.topk(cosine_scores, k=min(top_n_cv, len(job_corpus)))
                                display_recommendations(top_results, jobs_df)
                            else:
                                 st.warning(f"Could not extract text from {cv_file.name}.")
else:
    st.error("The recommendation engine could not be loaded. Please see the logs above.")
