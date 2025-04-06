import streamlit as st
st.title("CineSLEUTH")

import pandas as pd
from apyori import apriori
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
movies_file = './datasets/movies.csv'
ratings_file = './datasets/ratings.csv'
tags_file = './datasets/tag.csv'
genome_tags_file = './datasets/genome_tags.csv'

# Load datasets
@st.cache_data
def load_data():
    movies = pd.read_csv(movies_file)
    tags = pd.read_csv(tags_file)
    genome_tags = pd.read_csv(genome_tags_file)

    # Fill NaN values in the tags DataFrame
    tags['tag'] = tags['tag'].fillna('')

    # Merge tags with movies
    tagged_movies = pd.merge(tags, movies[['movieId', 'title']], on='movieId', how='inner')

    # Combine tags into a single string for each movie
    tagged_movies_grouped = tagged_movies.groupby('title')['tag'].apply(lambda x: ' '.join(str(tag) for tag in x if isinstance(tag, str))).reset_index()
    movies = pd.merge(movies, tagged_movies_grouped, on='title', how='left')

    return movies

movies = load_data()

# Create a TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
movies['tag'] = movies['tag'].fillna('')
movies['combined'] = movies['genres'] + ' ' + movies['tag']
tfidf_matrix = tfidf.fit_transform(movies['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on genres and tags
def get_recommendations(title, cosine_sim=cosine_sim):
    closest_match, score = process.extractOne(title, movies['title'].tolist())
    
    if score < 80:
        return "No close match found. Please check your spelling."
    
    idx = movies[movies['title'] == closest_match].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    
    # Display the similarity scores
    similarities = [i[1] for i in sim_scores]
    
    # Create a DataFrame for better visualization
    recommendations_df = pd.DataFrame({
        'Movie Title': movies['title'].iloc[movie_indices].tolist(),
        'Similarity Score': similarities
    })

    # Check for non-uniform similarity scores
    if len(set(similarities)) < 2:  # If all scores are the same
        st.warning("Warning: Similarity scores are uniform. Check the TF-IDF matrix.")
    
    # Plotting the similarity scores as a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(recommendations_df['Similarity Score'], labels=recommendations_df['Movie Title'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(recommendations_df)))
    plt.title('Top Movie Recommendations - Similarity Score Distribution')
    st.pyplot(plt)  # Use Streamlit to display the pie chart
    
    return recommendations_df

# User interaction and branding
st.title("CineSLEUTH: Your Ultimate Engine for Movie Recommendations")
st.subheader("Uncover Your Next Favorite Film with Smart Recommendations...")

# Initialize session state for input if not already set
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

# Input field for user movie title, pre-filled with session state value
user_input = st.text_input("Enter a movie title:", st.session_state['user_input']).strip()

# Autocomplete feature with clickable suggestions
if user_input:
    suggestions = process.extractBests(user_input, movies['title'].tolist(), limit=5)
    st.sidebar.subheader("Did you mean?")
    
    for i, (suggestion, score) in enumerate(suggestions):
        if st.sidebar.button(suggestion, key=f"suggestion_{i}"):  # Add unique keys to each button
            st.session_state['user_input'] = suggestion  # Update session state with the clicked suggestion
            recommendations = get_recommendations(suggestion)  # Automatically search for recommendations
            st.write(recommendations)  # Display the recommendations

# Get recommendations when user inputs a movie title manually
if st.button("SLEUTH"):
    if user_input:
        recommendations = get_recommendations(user_input)
        st.write(recommendations)
    else:
        st.error("Please enter a movie title.")

# Step 2: Process ratings in chunks for Apriori
chunk_size = 10000
transactions = []

# Read the ratings dataset in chunks
for chunk in pd.read_csv(ratings_file, chunksize=chunk_size):
    chunk_merged = pd.merge(chunk, movies[['movieId', 'title']], on='movieId')
    chunk_transactions = chunk_merged.groupby('userId')['title'].apply(list).values.tolist()
    transactions.extend(chunk_transactions)
    
    if len(transactions) > 1000000:
        rules = apriori(transactions, min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2)
        results = list(rules)
        st.write(f"Processed {len(transactions)} transactions with Apriori")
        transactions = []

# Final Apriori run after all chunks are processed
if transactions:
    rules = apriori(transactions, min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2)
    results = list(rules)

    st.subheader("Apriori Recommendations (Collaborative Filtering):")
    for item in results:
        pair = item[0]
        items = [x for x in pair]
        st.write(f"Rule: {items[0]} -> {items[1]}")
        st.write(f"Support: {item[1]}")
        st.write(f"Confidence: {item[2][0][2]}")
        st.write(f"Lift: {item[2][0][3]}")
        st.write("=====================================")
