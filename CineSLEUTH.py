import streamlit as st
import pandas as pd
from apyori import apriori
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
movies_file = 'datasets/movies.csv'
ratings_file = 'datasets/ratings.csv'
tags_file = 'datasets/tag.csv'
genome_tags_file = 'datasets/genome_tags.csv'

# Load datasets with error handling
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv(movies_file)
        tags = pd.read_csv(tags_file)
        genome_tags = pd.read_csv(genome_tags_file)
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}")
        st.stop()

    tags['tag'] = tags['tag'].fillna('')
    tagged_movies = pd.merge(tags, movies[['movieId', 'title']], on='movieId', how='inner')
    tagged_movies_grouped = tagged_movies.groupby('title')['tag'].apply(lambda x: ' '.join(str(tag) for tag in x if isinstance(tag, str))).reset_index()
    movies = pd.merge(movies, tagged_movies_grouped, on='title', how='left')
    return movies

movies = load_data()

# TF-IDF setup
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
movies['tag'] = movies['tag'].fillna('')
movies['combined'] = movies['genres'] + ' ' + movies['tag']
tfidf_matrix = tfidf.fit_transform(movies['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
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
    plt.pie(recommendations_df['Similarity Score'], labels=recommendations_df['Movie Title'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(recommendations_df)))
    plt.title('Top Movie Recommendations - Similarity Score Distribution')
    st.pyplot(plt)

    return recommendations_df

# UI Elements
st.title("🎬 CineSLEUTH: Your Ultimate Movie Recommender")
st.subheader("Uncover Your Next Favorite Film with Smart Recommendations")

# Input field and session state
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

user_input = st.text_input("Enter a movie title:", st.session_state['user_input']).strip()

# Autocomplete suggestions
if user_input:
    suggestions = process.extractBests(user_input, movies['title'].tolist(), limit=5)
    st.sidebar.subheader("Did you mean?")
    for i, (suggestion, score) in enumerate(suggestions):
        if st.sidebar.button(suggestion, key=f"suggestion_{i}"):
            st.session_state['user_input'] = suggestion
            recommendations = get_recommendations(suggestion)
            st.write(recommendations)

# Manual search
if st.button("🔍 SLEUTH"):
    if user_input:
        recommendations = get_recommendations(user_input)
        st.write(recommendations)
    else:
        st.error("Please enter a movie title.")

# Apriori wrapped in expander and button
with st.expander("📊 Run Apriori Collaborative Filtering (may take time)"):
    if st.button("Run Apriori"):
        st.info("Running Apriori on ratings data. Please wait...")

        transactions = []
        chunk_size = 10000
        for chunk in pd.read_csv(ratings_file, chunksize=chunk_size):
            chunk_merged = pd.merge(chunk, movies[['movieId', 'title']], on='movieId')
            chunk_transactions = chunk_merged.groupby('userId')['title'].apply(list).values.tolist()
            transactions.extend(chunk_transactions)

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

