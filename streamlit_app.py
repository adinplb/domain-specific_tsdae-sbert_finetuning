import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import logging

# --- 0. Setup Logging (Optional for Streamlit, but good for debugging) ---
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Configuration ---
# !!! IMPORTANT: Update this path to your saved fine-tuned model !!!
# This path is an example based on the training script's output pattern.
# Find the exact path in your 'output/' directory after running the training script.
DEFAULT_MODEL_PATH = "output/sbert_job_model_finetuned_sentence-transformers_all-MiniLM-L6-v2-YYYY-MM-DD_HH-MM-SS" # REPLACE WITH YOUR ACTUAL MODEL PATH

# Paths to your ORIGINAL CSV data files - NOW USING GITHUB RAW URLS AS DEFAULT
ORIGINAL_JOBS_CSV_FILE = "https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Filtered_Jobs_4000.csv"
ORIGINAL_ONET_CSV_FILE = "https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Occupation%20Data.csv"

# --- Helper function to process Filtered_Jobs_4000.csv ---
@st.cache_data # Cache the processed data
def process_jobs_csv_for_corpus(filepath_or_url):
    logger.info(f"Processing jobs CSV for corpus from: {filepath_or_url}")
    try:
        # Check if it's a URL or local path. os.path.exists won't work reliably for URLs.
        if not filepath_or_url.startswith(('http://', 'https://')) and not os.path.exists(filepath_or_url):
            st.error(f"Jobs CSV file not found at local path: {filepath_or_url}. Please ensure it's in the correct location or provide a valid URL.")
            return None, None
        jobs_df = pd.read_csv(filepath_or_url)
    except Exception as e:
        st.error(f"Error reading jobs CSV from {filepath_or_url}: {e}")
        return None, None

    columns_to_combine = [
        'Job.ID', 'Status', 'Title', 'Position', 'Company', 'City', 'State.Name',
        'Industry', 'Job.Description', 'Requirements', 'Salary', 'Employment.Type',
        'Education.Required'
    ]
    existing_columns = [col for col in columns_to_combine if col in jobs_df.columns]
    
    if not existing_columns:
        st.error("No specified columns for corpus found in the jobs CSV. Check column names.")
        return jobs_df, [] 

    logger.info(f"Combining columns for corpus: {existing_columns}")
    jobs_df_filled = jobs_df[existing_columns].fillna('').astype(str)
    processed_texts = jobs_df_filled.agg(' '.join, axis=1).tolist()
    cleaned_texts = [text.replace('\n', ' ').replace('\r', ' ') for text in processed_texts]
    
    logger.info(f"Processed {len(cleaned_texts)} job entries for corpus.")
    return jobs_df, cleaned_texts

# --- Load Model and Precompute Embeddings (Cached) ---
@st.cache_resource # Cache the loaded model
def load_model(model_path):
    logger.info(f"Loading fine-tuned model from: {model_path}")
    # This check is primarily for local paths
    if not model_path.startswith(('http://', 'https://')) and not os.path.exists(model_path):
        st.error(f"Model not found at local path: {model_path}. Please update DEFAULT_MODEL_PATH or the path in the sidebar.")
        return None
    try:
        model = SentenceTransformer(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

@st.cache_data # Cache the corpus embeddings
def encode_corpus(_model, corpus_texts): 
    if not corpus_texts or _model is None:
        return None
    logger.info(f"Encoding corpus of {len(corpus_texts)} documents... This may take a moment.")
    corpus_embeddings = _model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)
    logger.info("Corpus encoding complete.")
    return corpus_embeddings

# --- Load ONET Titles (Optional, Cached) ---
@st.cache_data
def load_onet_titles(filepath_or_url):
    logger.info(f"Loading ONET titles from: {filepath_or_url}")
    try:
        if not filepath_or_url.startswith(('http://', 'https://')) and not os.path.exists(filepath_or_url):
            logger.warning(f"ONET CSV file not found at local path: {filepath_or_url}. ONET comparison will be skipped.")
            return None
        onet_df = pd.read_csv(filepath_or_url)
        if 'Title' in onet_df.columns:
            return onet_df['Title'].dropna().unique().tolist()
        else:
            logger.warning("'Title' column not found in ONET CSV. Cannot load ONET titles.")
            return None
    except Exception as e:
        logger.error(f"Error reading ONET CSV from {filepath_or_url}: {e}")
        return None

# --- Main App ---
st.set_page_config(layout="wide")
st.title("✨ Job Recommendation Dashboard ✨")
st.write("Powered by a domain-adapted TSDAE-SBERT model.")

# --- Sidebar for Configuration ---
st.sidebar.header("Configuration")
model_path_input = st.sidebar.text_input("Enter path to your fine-tuned model:", DEFAULT_MODEL_PATH)
jobs_csv_path_input = st.sidebar.text_input("Path or URL to Filtered_Jobs_4000.csv:", ORIGINAL_JOBS_CSV_FILE)
onet_csv_path_input = st.sidebar.text_input("Path or URL to Occupation Data.csv (Optional):", ORIGINAL_ONET_CSV_FILE)

# --- Load resources ---
jobs_df_original, job_corpus_texts = process_jobs_csv_for_corpus(jobs_csv_path_input)
model = load_model(model_path_input)
onet_titles = load_onet_titles(onet_csv_path_input)

if model is None or jobs_df_original is None or not job_corpus_texts :
    st.warning("Model or job corpus could not be loaded. Please check paths and file contents. Dashboard functionality will be limited.")
else:
    corpus_embeddings = encode_corpus(model, job_corpus_texts)

    if corpus_embeddings is None:
        st.warning("Corpus embeddings could not be generated. Dashboard functionality will be limited.")
    else:
        st.sidebar.success(f"Model loaded and {len(job_corpus_texts)} jobs processed for recommendations.")

        st.header("🔍 Find Your Next Job")
        user_query = st.text_area("Describe your ideal job or paste a job description:", height=150, placeholder="e.g., python developer with machine learning skills, remote work preferred...")
        top_n = st.slider("Number of recommendations:", 1, 20, 10)

        if st.button("Get Recommendations", type="primary"):
            if not user_query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Finding the best matches..."):
                    query_embedding = model.encode(user_query, convert_to_tensor=True)

                    # Ensure embeddings are on the same device
                    target_device = query_embedding.device
                    corpus_embeddings_device = corpus_embeddings.to(target_device) if corpus_embeddings.device != target_device else corpus_embeddings
                        
                    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings_device)[0]
                    top_results = torch.topk(cosine_scores, k=min(top_n, len(job_corpus_texts)))

                    st.subheader(f"Top {len(top_results.values)} Job Recommendations:")
                    
                    results_data = []
                    for score, idx in zip(top_results.values, top_results.indices):
                        job_index = idx.item()
                        original_job_series = jobs_df_original.iloc[job_index]
                        corpus_text_snippet = job_corpus_texts[job_index][:200] + "..." if len(job_corpus_texts[job_index]) > 200 else job_corpus_texts[job_index]
                        
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

        # Optional: ONET Comparison Section
        if onet_titles and model: # Check if onet_titles were successfully loaded
            st.write("---")
            st.header("🆚 Compare with Standard ONET Titles (Examples)")
            if user_query.strip():
                # Ensure query_embedding is available. If button wasn't pressed, encode here.
                # Using a different variable name to avoid conflict if button was pressed
                onet_comparison_query_embedding = model.encode(user_query, convert_to_tensor=True)

                example_onet_titles = onet_titles[:5] 
                if example_onet_titles:
                    onet_data_for_table = []
                    with st.spinner("Calculating ONET title similarities..."):
                        for onet_title in example_onet_titles:
                            onet_embedding = model.encode(onet_title, convert_to_tensor=True)
                            
                            target_device = onet_comparison_query_embedding.device # Use current query embedding's device
                            onet_embedding_device = onet_embedding.to(target_device) if onet_embedding.device != target_device else onet_embedding
                                
                            sim_score = util.cos_sim(onet_comparison_query_embedding, onet_embedding_device).item()
                            onet_data_for_table.append({"ONET Title": onet_title, "Similarity to Query": f"{sim_score:.4f}"})
                    
                    if onet_data_for_table:
                        st.subheader("Similarity of your query to sample ONET titles:")
                        st.table(pd.DataFrame(onet_data_for_table))
            else:
                st.info("Enter a query above to see its similarity to ONET titles.")

st.sidebar.markdown("---")
st.sidebar.markdown("This dashboard demonstrates job recommendations using a Sentence Transformer model fine-tuned on job-specific data.")

