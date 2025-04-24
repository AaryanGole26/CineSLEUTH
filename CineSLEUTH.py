import streamlit as st
import pandas as pd
from apyori import apriori
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from supabase import create_client, Client
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = "datasets"  # Update if your bucket name differs
DATASET_FILES = {
    "movies": "movie.csv",
    "ratings": "ratings.csv",
    "tags": "tag.csv",
    "genome_tags": "genome_tags.csv"
}

# Validate environment variables
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("SUPABASE_URL or SUPABASE_KEY is missing")
    st.error("Supabase configuration is missing. Please set SUPABASE_URL and SUPABASE_KEY.")
    st.stop()

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized")
    # Debug: List available buckets
    buckets = supabase.storage.list_buckets()
    logger.info(f"Available buckets: {[bucket.name for bucket in buckets]}")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    st.error(f"Failed to initialize Supabase client: {e}")
    st.stop()

# Load datasets from Supabase
@st.cache_data
def load_data():
    try:
        logger.info("Fetching datasets from Supabase...")
        datasets = {}
        for key, file_name in DATASET_FILES.items():
            try:
                # Download file from Supabase bucket
                response = supabase.storage.from_(BUCKET_NAME).download(file_name)
                # Convert bytes to pandas DataFrame
                datasets[key] = pd.read_csv(io.BytesIO(response))
                logger.info(f"Loaded {file_name} from Supabase")
            except Exception as e:
                logger.error(f"Error downloading {file_name}: {e}")
                st.error(f"Error downloading {file_name}: {e}")
                st.stop()
        
        movies = datasets["movies"]
        tags = datasets["tags"]
        genome_tags = datasets["genome_tags"]

        # Preprocess data
        tags['tag'] = tags['tag'].fillna('')
        tagged_movies = pd.merge(tags, movies[['movieId', 'title']], on='movieId', how='inner')
        tagged_movies_grouped = tagged_movies.groupby('title')['tag'].apply(
            lambda x: ' '.join(str(tag) for tag in x if isinstance(tag, str))
        ).reset_index()
        movies = pd.merge(movies, tagged_movies_grouped, on='title', how='left')
        logger.info("Data preprocessing completed")
        return movies
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        st.error(f"Error parsing CSV file: {e}")
        st.stop()
    except Exception as e:
        logger.error(f"Unexpected error loading datasets: {e}")
        st.error(f"Unexpected error loading datasets: {e}")
        st.stop()

# Wrap startup operations
try:
    logger.info("Starting CineSLEUTH app")
    movies = load_data()

    # TF-IDF setup with max_features
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    movies['genres'] = movies['genres'].fillna('')
    movies['tag'] = movies['tag'].fillna('')
    movies['combined'] = movies['genres'] + ' ' + movies['tag']
    tfidf_matrix = tfidf.fit_transform(movies['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    logger.info("TF-IDF and cosine similarity computed")
except Exception as e:
    logger.error(f"Startup error: {e}")
    st.error(f"Startup error: {e}")
    st.stop()

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        closest_match, score = process.extractOne(title, movies['title'].tolist())
        if score < 80:
            return "No close match found. Please check your spelling."

        idx = movies[movies['title'] == closest_match].index[0]
        sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11]
        movie_indices = [i[0] for i in sim_scores]
        similarities = [i[1] for i in sim_scores]

        recommendations_df = pd.DataFrame({
            'Movie Title': movies['title'].iloc[movie_indices].tolist(),
            'Similarity Score': similarities
        })

        if len(set(similarities)) < 2:
            st.warning("Warning: Similarity scores are uniform. Check the TF-IDF matrix.")

        plt.clf()
        plt.figure(figsize=(8, 8))
        plt.pie(recommendations_df['Similarity Score'], labels=recommendations_df['Movie Title'],
                autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(recommendations_df)))
        plt.title('Top Movie Recommendations - Similarity Score Distribution')
        st.pyplot(plt)

        return recommendations_df
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        st.error(f"Error generating recommendations: {e}")
        return None

# UI Elements
st.title("🎬 CineSLEUTH: Your Ultimate Movie Recommender")
st.subheader("Uncover Your Next Favorite Film with Smart Recommendations")

# Input field and session state
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

user_input = st.text_input("Enter a movie title:", st.session_state['user_input']).strip()

# Autocomplete suggestions
if user_input:
    try:
        suggestions = process.extractBests(user_input, movies['title'].tolist(), limit=5)
        st.sidebar.subheader("Did you mean?")
        for i, (suggestion, score) in enumerate(suggestions):
            if st.sidebar.button(suggestion, key=f"suggestion_{i}"):
                st.session_state['user_input'] = suggestion
                recommendations = get_recommendations(suggestion)
                if recommendations is not None:
                    st.write(recommendations)
    except Exception as e:
        logger.error(f"Error in autocomplete suggestions: {e}")
        st.error(f"Error in autocomplete: {e}")

# Manual search
if st.button("🔍 SLEUTH"):
    if user_input:
        recommendations = get_recommendations(user_input)
        if recommendations is not None:
            st.write(recommendations)
    else:
        st.error("Please enter a movie title.")

# Apriori wrapped in expander and button
with st.expander("📊 Run Apriori Collaborative Filtering (may take time)"):
    if st.button("Run Apriori"):
        st.info("Running Apriori on ratings data. Please wait...")
        try:
            # Download ratings.csv from Supabase
            response = supabase.storage.from_(BUCKET_NAME).download(DATASET_FILES["ratings"])
            ratings_df = pd.read_csv(io.BytesIO(response))
            
            transactions = []
            chunk_size = 5000
            for chunk in pd.read_csv(io.BytesIO(response), chunksize=chunk_size):
                chunk_merged = pd.merge(chunk, movies[['movieId', 'title']], on='movieId')
                chunk_transactions = chunk_merged.groupby('userId')['title'].apply(list).values.tolist()
                transactions.extend(chunk_transactions[:100])
                if len(transactions) > 1000:
                    break

            rules = apriori(transactions, min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2)
            results = list(rules)

            if not results:
                st.warning("No strong association rules found.")
            else:
                st.subheader("Apriori Recommendations (Collaborative Filtering):")
                for item in results:
                    pair = item[0]
                    items = [x for x in pair]
                    st.write(f"Rule: {items[0]} -> {items[1]}")
                    st.write(f"Support: {item[1]:.4f}")
                    st.write(f"Confidence: {item[2][0][2]:.4f}")
                    st.write(f"Lift: {item[2][0][3]:.4f}")
                    st.write("------")
        except Exception as e:
            logger.error(f"Error in Apriori processing: {e}")
            st.error(f"Error running Apriori: {e}")
