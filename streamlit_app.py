import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, LoggingHandler, models, util, losses, InputExample
from sentence_transformers.datasets import DenoisingAutoEncoderDataset # Added this missing import
from torch.utils.data import DataLoader
import torch
import os
import logging
from datetime import datetime
import traceback

# --- 0. Setup Logging ---
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Configuration ---
# Path where the final fine-tuned model should be saved/loaded from.
DEFAULT_TRAINED_MODEL_OUTPUT_DIR = "trained_job_recommender_model" 

# Default paths to original CSV data files (can be URLs or local paths).
DEFAULT_JOBS_CSV_SOURCE = "https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Filtered_Jobs_4000.csv"
DEFAULT_ONET_CSV_SOURCE = "https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Occupation%20Data.csv"

# Base model for training.
BASE_MODEL_NAME_FOR_TRAINING = 'sentence-transformers/all-MiniLM-L6-v2'

# Training Hyperparameters.
TSDAE_EPOCHS = 1
TSDAE_BATCH_SIZE = 32
TSDAE_LEARNING_RATE = 3e-5
TSDAE_MAX_SEQ_LENGTH = 256

SBERT_EPOCHS = 1
SBERT_BATCH_SIZE = 16
SBERT_LEARNING_RATE = 2e-5

# Globals to store data to avoid re-processing.
job_corpus_texts_global = []
jobs_df_original_global = None

# --- Helper Functions for Data Processing ---
@st.cache_data
def process_jobs_csv_for_corpus(filepath_or_df):
    """
    Reads the jobs CSV, combines relevant text columns into single strings.
    This function is cached to avoid re-reading and re-processing the file on every script rerun.
    """
    global jobs_df_original_global # Declare that we will modify this global
    logger.info(f"Processing jobs data. Input type: {type(filepath_or_df)}")
    try:
        if isinstance(filepath_or_df, str):
            jobs_df = pd.read_csv(filepath_or_df)
        elif isinstance(filepath_or_df, pd.DataFrame):
            jobs_df = filepath_or_df.copy()
        else:
            logger.error("Invalid input for jobs data: Expected filepath string or pandas DataFrame.")
            return None, []
    except Exception as e:
        logger.error(f"Error processing jobs data source {filepath_or_df}: {e}")
        return None, []

    # Store the original dataframe in the global variable
    jobs_df_original_global = jobs_df.copy()

    columns_to_combine = [
        'Job.ID', 'Status', 'Title', 'Position', 'Company', 'City', 'State.Name',
        'Industry', 'Job.Description', 'Requirements', 'Salary', 'Employment.Type',
        'Education.Required'
    ]
    existing_columns = [col for col in columns_to_combine if col in jobs_df.columns]
    
    if not existing_columns:
        logger.error("No specified columns for corpus found in the jobs CSV/DataFrame.")
        return jobs_df.copy(), []

    logger.info(f"Combining columns for corpus: {existing_columns}")
    jobs_df_filled = jobs_df[existing_columns].fillna('').astype(str)
    processed_texts = jobs_df_filled.agg(' '.join, axis=1).tolist()
    cleaned_texts = [text.replace('\n', ' ').replace('\r', ' ') for text in processed_texts]
    
    logger.info(f"Processed {len(cleaned_texts)} job entries for corpus.")
    return jobs_df.copy(), cleaned_texts

def process_onet_csv_for_sbert_training(filepath_or_df):
    """
    Reads the ONET CSV and creates a list of InputExample objects for training.
    """
    examples = []
    try:
        if isinstance(filepath_or_df, str):
            onet_df = pd.read_csv(filepath_or_df)
        elif isinstance(filepath_or_df, pd.DataFrame):
            onet_df = filepath_or_df.copy()
        else:
            return []
    except Exception as e:
        logger.error(f"Error processing ONET data source {filepath_or_df}: {e}")
        return []

    if 'Title' in onet_df.columns and 'Description' in onet_df.columns:
        for _, row in onet_df.iterrows():
            examples.append(InputExample(texts=[str(row['Title']), str(row['Description'])], label=1.0))
    return examples

# --- Model Training Pipeline ---
def train_model_pipeline(jobs_data_src, onet_data_src, base_model, final_save_path):
    st.info(f"Starting model training pipeline. This will take a significant amount of time...")
    
    # Stage 1: TSDAE
    st.subheader("Stage 1: TSDAE Pre-training")
    # This call populates the global variables as a side effect.
    _, train_sentences_tsdae = process_jobs_csv_for_corpus(jobs_data_src)
    if not train_sentences_tsdae:
        st.error("TSDAE training failed: No job data processed.")
        return False

    with st.spinner("Training TSDAE model..."):
        word_embedding_model = models.Transformer(base_model, max_seq_length=TSDAE_MAX_SEQ_LENGTH)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        tsdae_train_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        tsdae_dataset = DenoisingAutoEncoderDataset(train_sentences_tsdae)
        tsdae_dataloader = DataLoader(tsdae_dataset, batch_size=TSDAE_BATCH_SIZE, shuffle=True)
        tsdae_loss = losses.DenoisingAutoEncoderLoss(model=tsdae_train_model, decoder_name_or_path=base_model, tie_encoder_decoder=True)
        
        tsdae_train_model.fit(
            train_objectives=[(tsdae_dataloader, tsdae_loss)], epochs=TSDAE_EPOCHS,
            weight_decay=0, scheduler='WarmupLinear', optimizer_params={'lr': TSDAE_LEARNING_RATE},
            warmup_steps=100, show_progress_bar=False, use_amp=True
        )
    st.write("TSDAE pre-training complete.")
    
    temp_tsdae_output_path = '/tmp/temp_tsdae_model'
    tsdae_train_model.save(temp_tsdae_output_path)
    
    # Stage 2: SBERT
    st.subheader("Stage 2: SBERT Fine-tuning")
    sbert_train_samples = process_onet_csv_for_sbert_training(onet_data_src)
    if not sbert_train_samples:
        st.error("SBERT fine-tuning failed: No ONET data processed.")
        return False
        
    with st.spinner("Fine-tuning SBERT model..."):
        sbert_model_to_finetune = SentenceTransformer(temp_tsdae_output_path)
        num_train_steps_sbert = len(sbert_train_samples) // SBERT_BATCH_SIZE * SBERT_EPOCHS
        sbert_warmup_steps = int(0.1 * num_train_steps_sbert)
        sbert_train_dataloader = DataLoader(sbert_train_samples, shuffle=True, batch_size=SBERT_BATCH_SIZE)
        sbert_loss = losses.MultipleNegativesRankingLoss(model=sbert_model_to_finetune)
        
        sbert_model_to_finetune.fit(
            train_objectives=[(sbert_train_dataloader, sbert_loss)], epochs=SBERT_EPOCHS,
            warmup_steps=sbert_warmup_steps, optimizer_params={'lr': SBERT_LEARNING_RATE},
            weight_decay=0.01, show_progress_bar=False, use_amp=True
        )
    
    os.makedirs(final_save_path, exist_ok=True)
    sbert_model_to_finetune.save(final_save_path)
    st.success(f"Model training complete! Fine-tuned model saved to: {final_save_path}")
    return True

# --- Load Model and Precompute Embeddings (Cached) ---
@st.cache_resource
def load_model(model_path):
    logger.info(f"Loading fine-tuned model from: {model_path}")
    if not os.path.exists(model_path):
        # This is expected if the model hasn't been trained yet, so we don't show an error here.
        logger.warning(f"Model not found at local path: {model_path}.")
        return None
    try:
        model = SentenceTransformer(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

@st.cache_data
def encode_corpus(_model, _corpus_texts_tuple):
    corpus_texts = list(_corpus_texts_tuple)
    logger.info(f"Encoding corpus of {len(corpus_texts)} documents...")
    return _model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=False)

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("✨ Job Recommendation Dashboard ✨")
st.write("Powered by a domain-adapted TSDAE-SBERT model.")

# Sidebar Configuration
st.sidebar.header("Setup & Configuration")
model_output_dir_input = st.sidebar.text_input("Trained Model Directory:", DEFAULT_TRAINED_MODEL_OUTPUT_DIR)
jobs_csv_source_input = st.sidebar.text_input("Jobs Data Source (Path or URL):", DEFAULT_JOBS_CSV_SOURCE)
onet_csv_source_input = st.sidebar.text_input("ONET Data Source (Path or URL, for training):", DEFAULT_ONET_CSV_SOURCE)

# Load data and model
# This populates jobs_df_original_global and job_corpus_texts_global
jobs_df_original_global, job_corpus_texts_global = process_jobs_csv_for_corpus(jobs_csv_source_input)
model = load_model(model_output_dir_input)

# Check if model exists, if not, offer to train it.
if model is None:
    st.sidebar.warning(f"Trained model not found at '{model_output_dir_input}'.")
    if st.sidebar.button("Train New Model (Time Consuming!)"):
        if job_corpus_texts_global:
            training_successful = train_model_pipeline(
                jobs_csv_source_input, onet_csv_source_input, 
                BASE_MODEL_NAME_FOR_TRAINING, model_output_dir_input
            )
            if training_successful:
                # Use st.experimental_rerun() to reload the app and load the new model
                st.experimental_rerun()
        else:
            st.sidebar.error("Cannot train model: Job data failed to load. Check Jobs Data Source.")

# Main dashboard area
if model and jobs_df_original_global is not None and job_corpus_texts_global:
    st.sidebar.success(f"Model loaded and {len(job_corpus_texts_global)} jobs ready for recommendations.")
    
    corpus_embeddings = encode_corpus(model, tuple(job_corpus_texts_global))
    
    if corpus_embeddings is not None:
        st.header("🔍 Find Your Next Job")
        user_query = st.text_area("Describe your ideal job or paste a job description:", height=150, placeholder="e.g., python developer with machine learning skills, remote work preferred...")
        top_n = st.slider("Number of recommendations:", 1, 20, 10)

        if st.button("Get Recommendations", type="primary"):
            if not user_query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Finding the best matches..."):
                    query_embedding = model.encode(user_query, convert_to_tensor=True)
                    target_device = query_embedding.device
                    corpus_embeddings_device = corpus_embeddings.to(target_device)
                    
                    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings_device)[0]
                    top_results = torch.topk(cosine_scores, k=min(top_n, len(job_corpus_texts_global)))

                    st.subheader(f"Top {len(top_results.values)} Job Recommendations:")
                    
                    if not top_results.values.numel():
                        st.info("No recommendations found based on your query.")
                    else:
                        for i, (score, idx) in enumerate(zip(top_results.values, top_results.indices)):
                            job_index = idx.item()
                            original_job_series = jobs_df_original_global.iloc[job_index]

                            with st.expander(f"**Rank {i+1}: {original_job_series.get('Title', 'N/A')}** at **{original_job_series.get('Company', 'N/A')}** (Score: {score.item():.4f})"):
                                st.markdown("---")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Position:** `{original_job_series.get('Position', 'N/A')}`")
                                    st.markdown(f"**Company:** `{original_job_series.get('Company', 'N/A')}`")
                                    st.markdown(f"**Location:** `{original_job_series.get('City', '')}, {original_job_series.get('State.Name', '')}`")
                                    st.markdown(f"**Job ID:** `{original_job_series.get('Job.ID', 'N/A')}`")
                                with col2:
                                    st.markdown(f"**Status:** `{original_job_series.get('Status', 'N/A')}`")
                                    st.markdown(f"**Employment Type:** `{original_job_series.get('Employment.Type', 'N/A')}`")
                                    st.markdown(f"**Education Required:** `{original_job_series.get('Education.Required', 'N/A')}`")
                                    st.markdown(f"**Industry:** `{str(original_job_series.get('Industry', 'N/A'))}`")
                                
                                st.markdown("##### Job Description")
                                st.text_area("Description", value=str(original_job_series.get('Job.Description', '')), height=200, disabled=True, label_visibility="collapsed", key=f"desc_{i}")
                                
                                requirements = str(original_job_series.get('Requirements', ''))
                                if requirements and requirements.lower() != 'nan' and requirements.strip():
                                    st.markdown("##### Requirements")
                                    st.text_area("Requirements", value=requirements, height=150, disabled=True, label_visibility="collapsed", key=f"req_{i}")
                                    
                                salary = str(original_job_series.get('Salary', ''))
                                if salary and salary.lower() != 'nan' and salary.strip():
                                     st.markdown(f"**Salary:** `{salary}`")

    else:
        st.error("Failed to generate corpus embeddings.")
else:
    if model is None:
        st.info("A trained model is required to proceed. Please check the model path in the sidebar or train a new model.")
    else:
        st.warning("Job corpus data could not be loaded. Please check the data source path in the sidebar.")

st.sidebar.info("This app demonstrates job recommendations. If a trained model isn't found, you can train one using the button above (this is a one-time, lengthy process).")

