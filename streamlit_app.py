import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, LoggingHandler, models, util, losses, InputExample
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
import torch
import os
import logging
from datetime import datetime
import traceback # For more detailed error logging

# --- 0. Setup Logging ---
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Configuration ---
# Path where the final fine-tuned model should be saved/loaded from
# The script will train and save the model here if it doesn't exist.
DEFAULT_TRAINED_MODEL_OUTPUT_DIR = "trained_job_recommender_model" 

# Default paths to original CSV data files (can be URLs or local paths)
DEFAULT_JOBS_CSV_SOURCE = "https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Filtered_Jobs_4000.csv"
DEFAULT_ONET_CSV_SOURCE = "https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Occupation%20Data.csv"

# Base model for training
BASE_MODEL_NAME_FOR_TRAINING = 'sentence-transformers/all-MiniLM-L6-v2'

# Training Hyperparameters (only used if the model needs to be trained)
TSDAE_EPOCHS = 1
TSDAE_BATCH_SIZE = 32
TSDAE_LEARNING_RATE = 3e-5
TSDAE_MAX_SEQ_LENGTH = 256

SBERT_EPOCHS = 1
SBERT_BATCH_SIZE = 16
SBERT_LEARNING_RATE = 2e-5

# Global to store the job corpus texts and original DataFrame from Filtered_Jobs_4000.csv
job_corpus_texts_global = []
jobs_df_original_global = None

# --- Helper Functions for Data Processing (from training script) ---
def process_jobs_csv_for_training_and_corpus(filepath_or_df):
    global job_corpus_texts_global, jobs_df_original_global
    logger.info(f"Processing jobs data. Input type: {type(filepath_or_df)}")
    try:
        if isinstance(filepath_or_df, str):
            logger.info(f"Reading jobs CSV from: {filepath_or_df}")
            jobs_df = pd.read_csv(filepath_or_df)
        elif isinstance(filepath_or_df, pd.DataFrame):
            logger.info("Using provided DataFrame for jobs data.")
            jobs_df = filepath_or_df.copy()
        else:
            logger.error("Invalid input for jobs data: Expected filepath string or pandas DataFrame.")
            st.error("Invalid input for jobs data source.")
            return None, [] # Return None for df, empty list for texts
    except FileNotFoundError:
        logger.error(f"Jobs CSV/DataFrame not found or accessible: {filepath_or_df}")
        st.error(f"Jobs CSV/DataFrame not found: {filepath_or_df}")
        return None, []
    except Exception as e:
        logger.error(f"Error processing jobs data source {filepath_or_df}: {e}")
        st.error(f"Error processing jobs data: {e}")
        # Reset globals on critical failure to ensure consistent state
        jobs_df_original_global = None
        job_corpus_texts_global = []
        return None, []

    jobs_df_original_global = jobs_df.copy() # Store the original DataFrame

    columns_to_combine = [
        'Job.ID', 'Status', 'Title', 'Position', 'Company', 'City', 'State.Name',
        'Industry', 'Job.Description', 'Requirements', 'Salary', 'Employment.Type',
        'Education.Required'
    ]
    existing_columns = [col for col in columns_to_combine if col in jobs_df.columns]
    if not existing_columns:
        logger.error("No specified columns for TSDAE found in the jobs CSV/DataFrame.")
        st.error("Required columns for processing jobs data are missing.")
        # Globals already assigned or will be empty
        return jobs_df_original_global, []

    logger.info(f"Combining columns for TSDAE: {existing_columns}")
    jobs_df_filled = jobs_df[existing_columns].fillna('').astype(str)
    processed_texts = jobs_df_filled.agg(' '.join, axis=1).tolist()
    cleaned_texts = [text.replace('\n', ' ').replace('\r', ' ') for text in processed_texts]
    
    job_corpus_texts_global = cleaned_texts # Populate the global corpus for recommendations
    logger.info(f"Processed {len(cleaned_texts)} job entries for TSDAE and stored for recommendations.")
    return jobs_df_original_global, cleaned_texts # Return original df and texts for TSDAE

def process_onet_csv_for_sbert_training(filepath_or_df):
    logger.info(f"Processing ONET data. Input type: {type(filepath_or_df)}")
    examples = []
    try:
        if isinstance(filepath_or_df, str):
            logger.info(f"Reading ONET CSV from: {filepath_or_df}")
            onet_df = pd.read_csv(filepath_or_df)
        elif isinstance(filepath_or_df, pd.DataFrame):
            logger.info("Using provided DataFrame for ONET data.")
            onet_df = filepath_or_df.copy()
        else:
            logger.error("Invalid input for ONET data: Expected filepath string or pandas DataFrame.")
            st.error("Invalid input for ONET data source.")
            return []
    except FileNotFoundError:
        logger.error(f"ONET CSV/DataFrame not found or accessible: {filepath_or_df}")
        st.error(f"ONET CSV/DataFrame not found: {filepath_or_df}")
        return []
    except Exception as e:
        logger.error(f"Error processing ONET data source {filepath_or_df}: {e}")
        st.error(f"Error processing ONET data: {e}")
        return []

    if 'Title' not in onet_df.columns or 'Description' not in onet_df.columns:
        logger.error("'Title' or 'Description' column not found in ONET CSV/DataFrame.")
        st.error("Required 'Title' or 'Description' columns missing in ONET data.")
        return []

    for index, row in onet_df.iterrows():
        title = str(row['Title']).replace('\n', ' ').replace('\r', ' ')
        description = str(row['Description']).replace('\n', ' ').replace('\r', ' ')
        examples.append(InputExample(texts=[title, description], label=1.0))
    logger.info(f"Processed {len(examples)} ONET entries for SBERT fine-tuning.")
    return examples

# --- Function to Train the Model (if needed) ---
def train_model_pipeline(jobs_data_src, onet_data_src, base_model_name_for_training, final_model_save_path):
    st.info(f"Starting model training pipeline. This will take a significant amount of time...")
    
    # Define specific output paths for intermediate models during training
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Ensure 'output' directory exists for temporary models
    os.makedirs("output", exist_ok=True)
    temp_tsdae_output_path = f'output/temp_tsdae_model_{base_model_name_for_training.replace("/", "_")}-{timestamp}'
    
    # --- Stage 1: TSDAE Pre-training ---
    st.subheader("Stage 1: TSDAE Pre-training")
    logger.info("--- Starting Stage 1: TSDAE Pre-training ---")
    
    processed_jobs_df, train_sentences_tsdae = process_jobs_csv_for_training_and_corpus(jobs_data_src)

    if not train_sentences_tsdae or processed_jobs_df is None:
        st.error("TSDAE training failed: No job data processed or error during processing.")
        logger.error("No data available for TSDAE pre-training during pipeline.")
        return False

    with st.spinner("Defining TSDAE model..."):
        logger.info(f"Defining SentenceTransformer model for TSDAE with base: {base_model_name_for_training}")
        word_embedding_model = models.Transformer(base_model_name_for_training, max_seq_length=TSDAE_MAX_SEQ_LENGTH)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        tsdae_train_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        tsdae_dataset = DenoisingAutoEncoderDataset(train_sentences_tsdae)
        tsdae_dataloader = DataLoader(tsdae_dataset, batch_size=TSDAE_BATCH_SIZE, shuffle=True)
        tsdae_loss = losses.DenoisingAutoEncoderLoss(model=tsdae_train_model, decoder_name_or_path=base_model_name_for_training, tie_encoder_decoder=True)
    st.write("TSDAE model defined.")

    with st.spinner(f"Training TSDAE model for {TSDAE_EPOCHS} epoch(s)... (This is time-consuming)"):
        logger.info("Starting TSDAE model training...")
        tsdae_train_model.fit(
            train_objectives=[(tsdae_dataloader, tsdae_loss)], epochs=TSDAE_EPOCHS,
            weight_decay=0, scheduler='WarmupLinear', optimizer_params={'lr': TSDAE_LEARNING_RATE},
            warmup_steps=100, 
            show_progress_bar=False, # Disabled for Streamlit compatibility
            use_amp=True
        )
    st.write("TSDAE pre-training complete.")
    
    os.makedirs(temp_tsdae_output_path, exist_ok=True)
    tsdae_train_model.save(temp_tsdae_output_path)
    logger.info(f"TSDAE pre-trained model saved to temporary path: {temp_tsdae_output_path}")

    # --- Stage 2: SBERT Fine-tuning on ONET Data ---
    st.subheader("Stage 2: SBERT Fine-tuning")
    logger.info("--- Starting Stage 2: SBERT Fine-tuning on ONET Data ---")
    
    with st.spinner("Loading TSDAE model and preparing SBERT data..."):
        sbert_model_to_finetune = SentenceTransformer(temp_tsdae_output_path)
        sbert_train_samples = process_onet_csv_for_sbert_training(onet_data_src)

    if not sbert_train_samples:
        st.error("SBERT fine-tuning failed: No ONET data processed.")
        logger.error("No data available for SBERT fine-tuning during pipeline.")
        return False

    num_train_steps_sbert = len(sbert_train_samples) // SBERT_BATCH_SIZE * SBERT_EPOCHS
    sbert_warmup_steps = int(0.1 * num_train_steps_sbert) if num_train_steps_sbert > 0 else 0

    with st.spinner("Defining SBERT loss and dataloader..."):
        sbert_train_dataloader_mnrl = DataLoader(sbert_train_samples, shuffle=True, batch_size=SBERT_BATCH_SIZE)
        sbert_loss_mnrl = losses.MultipleNegativesRankingLoss(model=sbert_model_to_finetune)
    st.write("SBERT components defined.")

    with st.spinner(f"Fine-tuning SBERT model for {SBERT_EPOCHS} epoch(s)... (This is time-consuming)"):
        logger.info("Starting SBERT model fine-tuning...")
        sbert_model_to_finetune.fit(
            train_objectives=[(sbert_train_dataloader_mnrl, sbert_loss_mnrl)], epochs=SBERT_EPOCHS,
            warmup_steps=sbert_warmup_steps, optimizer_params={'lr': SBERT_LEARNING_RATE},
            weight_decay=0.01, 
            show_progress_bar=False, # Disabled for Streamlit compatibility
            use_amp=True, save_best_model=False 
        )
    
    os.makedirs(final_model_save_path, exist_ok=True)
    sbert_model_to_finetune.save(final_model_save_path)
    logger.info(f"SBERT fine-tuned model saved to: {final_model_save_path}")
    st.success(f"Model training complete! Fine-tuned model saved to: {final_model_save_path}")
    return True


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("✨ Job Recommendation Dashboard ✨")
st.write("Powered by a domain-adapted TSDAE-SBERT model.")

# Sidebar for configuration
st.sidebar.header("Setup & Configuration")
model_output_dir_input = st.sidebar.text_input("Trained Model Directory:", DEFAULT_TRAINED_MODEL_OUTPUT_DIR)
jobs_csv_source_input = st.sidebar.text_input("Jobs Data Source (Path or URL):", DEFAULT_JOBS_CSV_SOURCE)
onet_csv_source_input = st.sidebar.text_input("ONET Data Source (Path or URL, for training):", DEFAULT_ONET_CSV_SOURCE)

model_file_check_path = os.path.join(model_output_dir_input, "pytorch_model.bin")

# Attempt to load job corpus early, as it's needed whether model is loaded or trained
if not job_corpus_texts_global: 
    process_jobs_csv_for_training_and_corpus(jobs_csv_source_input)


# Load or Train Model
model = None
if os.path.exists(model_file_check_path):
    with st.spinner(f"Loading existing fine-tuned model from: {model_output_dir_input}"):
        try:
            model = SentenceTransformer(model_output_dir_input)
            st.sidebar.success("Pre-trained model loaded successfully!")
            if not job_corpus_texts_global: 
                 st.sidebar.error("Job corpus could not be loaded even though model exists. Check Jobs Data Source and refresh.")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}. Training may be required.")
            logger.error(f"Error loading existing model: {e}\n{traceback.format_exc()}")
            model = None 
else:
    st.sidebar.warning(f"Trained model not found at '{model_output_dir_input}'.")
    if st.sidebar.button("Train New Model (Time Consuming!)"):
        if not job_corpus_texts_global :
            st.sidebar.info("Attempting to load job data before training...")
            process_jobs_csv_for_training_and_corpus(jobs_csv_source_input) 
        
        if not job_corpus_texts_global : 
            st.sidebar.error("Cannot train model: Jobs data (corpus) failed to load. Check Jobs Data Source in sidebar.")
        else:
            try:
                training_successful = train_model_pipeline(
                    jobs_csv_source_input, 
                    onet_csv_source_input, 
                    BASE_MODEL_NAME_FOR_TRAINING, 
                    model_output_dir_input 
                )
                if training_successful:
                    try:
                        model = SentenceTransformer(model_output_dir_input) 
                        st.sidebar.success("Model trained and loaded successfully!")
                    except Exception as e_load:
                         st.sidebar.error(f"Error loading newly trained model: {e_load}")
                         logger.error(f"Error loading newly trained model: {e_load}\n{traceback.format_exc()}")
                         model = None
                else:
                    st.sidebar.error("Model training failed.")
                    model = None
            except Exception as e_train_pipe:
                st.sidebar.error(f"Error during training pipeline: {e_train_pipe}")
                logger.error(f"Training pipeline error: {e_train_pipe}\n{traceback.format_exc()}")
                model = None


# --- Main Dashboard Area (Only if model and corpus are available) ---
if model and job_corpus_texts_global and jobs_df_original_global is not None:
    st.sidebar.info(f"Ready for recommendations with {len(job_corpus_texts_global)} jobs.")
    
    @st.cache_data 
    def get_or_encode_corpus(_model_path_for_cache_key, _corpus_texts_for_cache_key_tuple):
        _corpus_texts_list = list(_corpus_texts_for_cache_key_tuple)
        try:
            loaded_model_for_encoding = SentenceTransformer(_model_path_for_cache_key)
            logger.info(f"Encoding corpus of {len(_corpus_texts_list)} documents for dashboard...")
            corpus_embeddings = loaded_model_for_encoding.encode(
                _corpus_texts_list, 
                convert_to_tensor=True, 
                show_progress_bar=False 
            )
            logger.info("Dashboard corpus encoding complete.")
            return corpus_embeddings
        except Exception as e:
            st.error(f"Error encoding corpus within cached function: {e}")
            logger.error(f"Cached corpus encoding error: {e}\n{traceback.format_exc()}")
            return None

    corpus_embeddings = get_or_encode_corpus(model_output_dir_input, tuple(job_corpus_texts_global))

    if corpus_embeddings is not None:
        st.header("🔍 Find Your Next Job")
        user_query = st.text_area("Describe your ideal job or paste a job description:", height=150, placeholder="e.g., python developer with machine learning skills, remote work preferred...")
        top_n = st.slider("Number of recommendations:", 1, 20, 10)

        if st.button("Get Recommendations", type="primary"):
            if not user_query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Finding the best matches..."):
                    try:
                        query_embedding = model.encode(user_query, convert_to_tensor=True)
                        target_device = query_embedding.device
                        corpus_embeddings_device = corpus_embeddings.to(target_device) if corpus_embeddings.device != target_device else corpus_embeddings
                        
                        cosine_scores = util.cos_sim(query_embedding, corpus_embeddings_device)[0]
                        top_results = torch.topk(cosine_scores, k=min(top_n, len(job_corpus_texts_global)))

                        st.subheader(f"Top {len(top_results.values)} Job Recommendations:")
                        
                        if not top_results.values.numel():
                            st.info("No recommendations found based on your query.")
                        else:
                            for i, (score, idx) in enumerate(zip(top_results.values, top_results.indices)):
                                job_index = idx.item()
                                original_job_series = jobs_df_original_global.iloc[job_index]

                                title = original_job_series.get('Title', 'N/A')
                                company = original_job_series.get('Company', 'N/A')
                                
                                with st.expander(f"**Rank {i+1}: {title}** at **{company}** (Score: {score.item():.4f})"):
                                    # Use two columns for a cleaner layout
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown(f"**Position:** {original_job_series.get('Position', 'N/A')}")
                                        st.markdown(f"**Company:** {company}")
                                        st.markdown(f"**Location:** {original_job_series.get('City', '')}, {original_job_series.get('State.Name', '')}")
                                        st.markdown(f"**Job ID:** {original_job_series.get('Job.ID', 'N/A')}")
                                    with col2:
                                        st.markdown(f"**Status:** {original_job_series.get('Status', 'N/A')}")
                                        st.markdown(f"**Employment Type:** {original_job_series.get('Employment.Type', 'N/A')}")
                                        st.markdown(f"**Education Required:** {original_job_series.get('Education.Required', 'N/A')}")
                                        st.markdown(f"**Industry:** {str(original_job_series.get('Industry', 'N/A'))}")
                                    
                                    st.markdown("---")
                                    st.markdown("#### Job Description")
                                    # Use a text area for the long description to make it scrollable
                                    st.text_area("Description", value=str(original_job_series.get('Job.Description', '')), height=150, disabled=True, label_visibility="collapsed")
                                    
                                    # Only show Requirements and Salary if they exist
                                    requirements = str(original_job_series.get('Requirements', ''))
                                    if requirements and requirements.lower() != 'nan':
                                        st.markdown("---")
                                        st.markdown("#### Requirements")
                                        st.text_area("Requirements", value=requirements, height=100, disabled=True, label_visibility="collapsed")
                                        
                                    salary = str(original_job_series.get('Salary', ''))
                                    if salary and salary.lower() != 'nan':
                                         st.markdown("---")
                                         st.markdown(f"**Salary:** {salary}")

                    except Exception as e:
                        st.error(f"Error during recommendation: {e}")
                        logger.error(f"Recommendation error: {e}")
                        logger.error(traceback.format_exc())
        
        # Optional ONET Comparison section (rest of the code is the same)
        # ...
    else:
        st.info("Corpus embeddings are not available. Please ensure data is loaded and processed, and the model is available.")
elif not os.path.exists(model_file_check_path):
    st.info("Model not found. Please use the sidebar to train a new model if data sources are correctly specified. Ensure data sources are valid before training.")
else: 
    st.warning("Dashboard cannot proceed. Model may exist but data (job corpus) could not be loaded. Check 'Jobs Data Source' in the sidebar and ensure the file is accessible and correctly formatted. Check logs for details.")


st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates job recommendations. If a trained model isn't found, you can train one using the button above (this is a one-time, lengthy process).")

