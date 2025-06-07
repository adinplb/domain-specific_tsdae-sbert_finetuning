import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# --- Page Configuration ---
st.set_page_config(
    page_title="Job Recommendation Engine",
    page_icon="🤖",
    layout="wide",
)

# --- Configuration ---
# We will load a pre-trained model directly from the Hugging Face Hub.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
JOBS_DATA_URL = "https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Filtered_Jobs_4000.csv"

# --- Caching Functions for Performance ---

@st.cache_resource
def load_model(model_name):
    """
    Loads a SentenceTransformer model from the Hugging Face Hub.
    Using st.cache_resource ensures the model is loaded only once.
    """
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading the model '{model_name}': {e}")
        return None

@st.cache_data
def load_and_process_job_data(url):
    """
    Loads job data from a URL and prepares the text corpus for embedding.
    Using st.cache_data ensures the data is loaded and processed only once.
    """
    try:
        jobs_df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Failed to load data from URL: {e}")
        return None, []

    # Define columns to be combined for the embedding corpus
    columns_to_combine = [
        "Title",
        "Position",
        "Company",
        "City",
        "State.Name",
        "Industry",
        "Job.Description",
        "Requirements",
        "Employment.Type",
        "Education.Required",
    ]
    # Use only the columns that actually exist in the DataFrame
    existing_columns = [col for col in columns_to_combine if col in jobs_df.columns]
    
    jobs_df_filled = jobs_df[existing_columns].fillna("").astype(str)
    # Combine columns into a single string for each job
    processed_texts = jobs_df_filled.agg(" ".join, axis=1).tolist()
    # Clean text by removing newline characters
    cleaned_texts = [
        text.replace("\n", " ").replace("\r", " ") for text in processed_texts
    ]
    
    return jobs_df.copy(), cleaned_texts

@st.cache_data
def encode_corpus(_model, job_corpus_texts):
    """
    Encodes the job corpus using the provided model.
    The result is cached to avoid re-calculating embeddings on every interaction.
    """
    if not job_corpus_texts or _model is None:
        return None
    with st.spinner("Encoding the job database... This may take a moment on first run."):
        # Encode the text to create vector embeddings
        corpus_embeddings = _model.encode(
            job_corpus_texts, convert_to_tensor=True, show_progress_bar=False
        )
    return corpus_embeddings


# --- Streamlit UI ---
st.title("📄 Job Recommendation Engine")
st.markdown(
    "Enter your resume summary, skills, or a description of your ideal job below. The engine will compare your query against a database of 4,000 job descriptions to find the most relevant matches."
)

# --- Load Model and Data ---
model = load_model(MODEL_NAME)
if model:
    jobs_df, job_corpus = load_and_process_job_data(JOBS_DATA_URL)
    
    if jobs_df is not None and job_corpus:
        corpus_embeddings = encode_corpus(model, job_corpus)

        # --- User Input ---
        st.subheader("Your Job Query")
        user_query = st.text_area(
            "Enter your query here:",
            "Seeking a senior software engineer role specializing in backend development with Python, Django, and cloud services like AWS.",
            height=150,
            label_visibility="collapsed"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
             top_n = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10)
        with col2:
            st.write("") # Spacer
            st.write("") # Spacer
            find_button = st.button("✨ Find My Dream Job", use_container_width=True)

        if find_button:
            if not user_query.strip():
                st.warning("Please enter a query.")
            elif corpus_embeddings is None:
                st.error("Could not generate corpus embeddings. Please check the data source.")
            else:
                with st.spinner("Searching for the best matches..."):
                    # Encode the user query into a vector
                    query_embedding = model.encode(user_query, convert_to_tensor=True)

                    # Compute cosine similarity between the query and all job descriptions
                    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                    
                    # Get the top N results
                    top_results = torch.topk(cosine_scores, k=min(top_n, len(job_corpus)))

                    st.success(f"Here are your top {top_n} job recommendations:")

                    # NEW: Display results in an expander format
                    for i, (score, idx) in enumerate(zip(top_results.values, top_results.indices)):
                        job_index = idx.item()
                        original_job = jobs_df.iloc[job_index]
                        
                        # Prepare the title for the expander
                        expander_title = f"**{i+1}. {original_job.get('Title', 'N/A')}** at **{original_job.get('Company', 'N/A')}** (Score: {score.item():.2f})"
                        
                        with st.expander(expander_title):
                            # Create two columns for a cleaner layout
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

else:
    st.error("The recommendation engine is currently unavailable. The model could not be loaded from Hugging Face Hub.")
