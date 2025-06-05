import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, LoggingHandler, models, util, losses, InputExample
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
import torch
import os
import logging
from datetime import datetime

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
    
    # Data for TSDAE. process_jobs_csv_for_training_and_corpus populates globals.
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
            warmup_steps=100, show_progress_bar=True, use_amp=True
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
            weight_decay=0.01, # output_path=final_model_save_path, # Removed to save explicitly after fit
            show_progress_bar=True, use_amp=True, save_best_model=False 
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
# This also populates jobs_df_original_global
# Do this only once at the start, or if jobs_csv_source_input changes (more complex for now)
if not job_corpus_texts_global: # Initialize if empty
    process_jobs_csv_for_training_and_corpus(jobs_csv_source_input)


# Load or Train Model
model = None
if os.path.exists(model_file_check_path):
    with st.spinner(f"Loading existing fine-tuned model from: {model_output_dir_input}"):
        try:
            model = SentenceTransformer(model_output_dir_input)
            st.sidebar.success("Pre-trained model loaded successfully!")
            # Job corpus should already be loaded by the call above
            if not job_corpus_texts_global: # Double check if corpus loading failed for some reason
                 st.sidebar.error("Job corpus could not be loaded even though model exists. Check Jobs Data Source and refresh.")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}. Training may be required.")
            model = None 
else:
    st.sidebar.warning(f"Trained model not found at '{model_output_dir_input}'.")
    if st.sidebar.button("Train New Model (Time Consuming!)"):
        # Ensure corpus is loaded before attempting to train
        if not job_corpus_texts_global:
            # Attempt to load corpus again if it failed earlier, using current sidebar input
            st.sidebar.info("Attempting to load job data before training...")
            process_jobs_csv_for_training_and_corpus(jobs_csv_source_input) 
        
        if not job_corpus_texts_global : # Check again after attempt
            st.sidebar.error("Cannot train model: Jobs data (corpus) failed to load. Check Jobs Data Source in sidebar.")
        else:
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
                except Exception as e:
                     st.sidebar.error(f"Error loading newly trained model: {e}")
                     model = None
            else:
                st.sidebar.error("Model training failed.")
                model = None


# --- Main Dashboard Area (Only if model and corpus are available) ---
if model and job_corpus_texts_global and jobs_df_original_global is not None:
    st.sidebar.info(f"Ready for recommendations with {len(job_corpus_texts_global)} jobs.")
    
    @st.cache_data 
    def get_or_encode_corpus(_model_path_for_cache_key, _corpus_texts_for_cache_key_tuple):
        _corpus_texts_list = list(_corpus_texts_for_cache_key_tuple)
        loaded_model_for_encoding = SentenceTransformer(_model_path_for_cache_key)
        logger.info(f"Encoding corpus of {len(_corpus_texts_list)} documents for dashboard...")
        corpus_embeddings = loaded_model_for_encoding.encode(
            _corpus_texts_list, 
            convert_to_tensor=True, 
            show_progress_bar=False # Disabled for potentially better Streamlit compatibility
        )
        logger.info("Dashboard corpus encoding complete.")
        return corpus_embeddings

    try:
        corpus_embeddings = get_or_encode_corpus(model_output_dir_input, tuple(job_corpus_texts_global))
    except Exception as e:
        st.error(f"Error encoding corpus for dashboard: {e}. Check model path and data.")
        logger.error(f"Dashboard corpus encoding error: {e}\n{traceback.format_exc()}")
        corpus_embeddings = None


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
                        results_data = []
                        for score, idx in zip(top_results.values, top_results.indices):
                            job_index = idx.item()
                            original_job_series = jobs_df_original_global.iloc[job_index]
                            corpus_text_snippet = job_corpus_texts_global[job_index][:200] + "..." if len(job_corpus_texts_global[job_index]) > 200 else job_corpus_texts_global[job_index]
                            
                            results_data.append({
                                "Rank": len(results_data) + 1,
                                "Similarity Score": f"{score.item():.4f}",
                                "Job ID": original_job_series.get('Job.ID', 'N/A'),
                                "Title": original_job_series.get('Title', 'N/A'),
                                "Company": original_job_series.get('Company', 'N/A'),
                                "Location": f"{original_job_series.get('City', '')}, {original_job_series.get('State.Name', '')}",
                                "Corpus Text Snippet": corpus_text_snippet
                            })
                        
                        if results_data:
                            st.dataframe(pd.DataFrame(results_data), use_container_width=True)
                        else:
                            st.info("No recommendations found based on your query.")
                    except Exception as e:
                        st.error(f"Error during recommendation: {e}")
                        logger.error(f"Recommendation error: {e}")
                        import traceback # Make sure traceback is imported if used here
                        logger.error(traceback.format_exc())
        
        # Optional ONET Comparison
        if DEFAULT_ONET_CSV_SOURCE: 
            onet_titles_for_comparison = None
            try:
                onet_df_comp = pd.read_csv(onet_csv_source_input) 
                if 'Title' in onet_df_comp.columns:
                    onet_titles_for_comparison = onet_df_comp['Title'].dropna().unique().tolist()
            except Exception as e:
                logger.warning(f"Could not load ONET data for comparison from {onet_csv_source_input}: {e}")

            if onet_titles_for_comparison:
                st.write("---")
                st.header("🆚 Compare with Standard ONET Titles (Examples)")
                if user_query.strip():
                    try:
                        current_query_embedding = model.encode(user_query, convert_to_tensor=True)
                        example_onet_titles = onet_titles_for_comparison[:5]
                        if example_onet_titles:
                            onet_data_for_table = []
                            with st.spinner("Calculating ONET title similarities..."):
                                for onet_title in example_onet_titles:
                                    onet_embedding = model.encode(onet_title, convert_to_tensor=True)
                                    target_device = current_query_embedding.device
                                    onet_embedding_device = onet_embedding.to(target_device) if onet_embedding.device != target_device else onet_embedding
                                    sim_score = util.cos_sim(current_query_embedding, onet_embedding_device).item()
                                    onet_data_for_table.append({"ONET Title": onet_title, "Similarity to Query": f"{sim_score:.4f}"})
                            if onet_data_for_table:
                                st.subheader("Similarity of your query to sample ONET titles:")
                                st.table(pd.DataFrame(onet_data_for_table))
                    except Exception as e:
                        st.error(f"Error during ONET comparison: {e}")
                        logger.error(f"ONET comparison error: {e}")
                        import traceback # Ensure traceback is imported
                        logger.error(traceback.format_exc())

                else:
                    st.info("Enter a query above to see its similarity to ONET titles.")
    else:
        st.info("Corpus embeddings are not available. Please ensure data is loaded and processed, and the model is available.")
elif not os.path.exists(model_file_check_path):
    st.info("Model not found. Please use the sidebar to train a new model if data sources are correctly specified. Ensure data sources are valid before training.")
else: 
    st.warning("Dashboard cannot proceed. Model may exist but data (job corpus) could not be loaded. Check 'Jobs Data Source' in the sidebar and ensure the file is accessible and correctly formatted. Check logs for details.")


st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates job recommendations. If a trained model isn't found, you can train one using the button above (this is a one-time, lengthy process).")

