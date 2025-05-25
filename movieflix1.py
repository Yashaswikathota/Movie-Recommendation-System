import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
import re
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from urllib.parse import quote
import datetime

# --- Page config ---
st.set_page_config(layout="wide", page_title="🎮 Hybrid Movie Recommender - Mew Edition")

# --- NLTK Setup ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# --- TMDb API Key ---
TMDB_API_KEY = "eb9c4092d8db64a61384a5d157679d6b"  # Insert your API key here

def get_tmdb_poster(movie_id):
    if not TMDB_API_KEY.strip():
        return None
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

def get_youtube_link(movie_title):
    query = quote(f"{movie_title} trailer")
    return f"https://www.youtube.com/results?search_query={query}"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

@st.cache_data
def load_data():
    movies_df = pd.read_csv('movies.csv')
    credits_df = pd.read_csv('credits.csv')

    movies = movies_df.merge(credits_df, on='title')
    movies.dropna(subset=['overview', 'genres', 'keywords', 'cast', 'crew'], inplace=True)
    movies.reset_index(drop=True, inplace=True)
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies.dropna(subset=['release_date'], inplace=True)

    def extract_genres(text):
        try:
            return [g['name'] for g in ast.literal_eval(text)]
        except:
            return []

    def extract_keywords(text):
        try:
            return [kw['name'].replace(" ", "") for kw in ast.literal_eval(text)]
        except:
            return []

    def extract_actors(text):
        try:
            return [actor['name'].replace(" ", "") for actor in ast.literal_eval(text)[:3]]
        except:
            return []

    def extract_director(text):
        try:
            crew = ast.literal_eval(text)
            for member in crew:
                if member['job'] == 'Director':
                    return member['name'].replace(" ", "")
        except:
            return ''

    movies['genre_list'] = movies['genres'].apply(extract_genres)
    movies['keywords_list'] = movies['keywords'].apply(extract_keywords)
    movies['actors'] = movies['cast'].apply(extract_actors)
    movies['director'] = movies['crew'].apply(extract_director)

    movies['overview_tokens'] = movies['overview'].apply(lambda x: x.split() if pd.notna(x) else [])
    movies['tags'] = movies['overview_tokens'] + movies['genre_list'] + movies['keywords_list'] + movies['actors'] + movies['director'].apply(lambda x: [x] if x else [])
    movies['tags'] = movies['tags'].apply(lambda x: " ".join([str(item).lower() for item in x if item and str(item).lower() not in stop_words]))
    movies['cleaned_tags'] = movies['tags'].apply(clean_text)

    return movies

movies = load_data()

@st.cache_data
def fit_knn(movies):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['cleaned_tags'])
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)
    return tfidf_matrix, knn_model

tfidf_matrix, knn_model = fit_knn(movies)

def predict_rating(user_id, movie_id):
    np.random.seed(user_id + movie_id)
    return np.random.uniform(2.5, 4.5)

def compute_hybrid_score(sim_score, cf_score, content_weight, cf_weight):
    normalized_cf = cf_score / 5.0 if cf_score else 0
    return (content_weight * sim_score) + (cf_weight * normalized_cf)

def hybrid_recommendation(user_id, movie_title, top_n=5, content_weight=0.5, cf_weight=0.5, selected_genres=None):
    if movie_title not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie_title].index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:30]

    hybrid_scores = []

    for movie_idx, sim_score in sim_scores:
        movie = movies.iloc[movie_idx]
        movie_id = movie['id']
        cf_score = predict_rating(user_id, movie_id)
        combined_score = compute_hybrid_score(sim_score, cf_score, content_weight, cf_weight)

        if selected_genres and not any(g in selected_genres for g in movie['genre_list']):
            continue

        poster_url = get_tmdb_poster(movie_id)

        hybrid_scores.append({
            'title': movie['title'],
            'genres': ", ".join(movie['genre_list']) if isinstance(movie['genre_list'], list) else '',
            'poster_url': poster_url,
            'rating': cf_score
        })

    sorted_scores = sorted(hybrid_scores, key=lambda x: x['rating'], reverse=True)[:top_n]
    return sorted_scores

def get_top_10_movies_overall(movies_df):
    top_movies = movies_df.sort_values(by='vote_average', ascending=False).head(10)
    top_list = []
    for _, row in top_movies.iterrows():
        poster_url = get_tmdb_poster(row['id'])
        top_list.append({
            'title': row['title'],
            'poster_url': poster_url,
            'rating': row['vote_average']
        })
    return top_list

def get_trending_now(movies_df, top_n=10):
    current_date = datetime.datetime.now()
    cutoff_date = current_date - pd.DateOffset(years=2)
    recent_movies = movies_df[movies_df['release_date'] >= cutoff_date]
    trending = recent_movies.sort_values(by=['vote_count', 'vote_average'], ascending=False).head(top_n)

    trending_list = []
    for _, row in trending.iterrows():
        poster_url = get_tmdb_poster(row['id'])
        trending_list.append({
            'title': row['title'],
            'poster_url': poster_url,
            'rating': row['vote_average']
        })
    return trending_list

def get_stars(rating):
    stars = "⭐" * int(round(rating))
    return stars

# --- CSS Styling ---
st.markdown("""
    <style>
    body {
        background-color: #0d1117;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }

    .stApp {
        background-color: #0d1117;
    }

    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.6);
        transition: transform 0.3s ease;
    }

    .stImage > img:hover {
        transform: scale(1.05);
    }

    .movie-title {
        font-weight: bold;
        font-size: 18px;
        color: #ffb700;
        margin-top: 5px;
        text-shadow: 1px 1px 3px black;
    }

    .genre-badge {
        display: inline-block;
        background-color: #1f6feb;
        color: white;
        padding: 4px 10px;
        margin: 2px;
        border-radius: 15px;
        font-size: 11px;
    }

    .trailer-link a {
        color: #58a6ff;
        text-decoration: none;
        font-size: 13px;
    }

    .trailer-link a:hover {
        text-decoration: underline;
    }

    .stNumberInput > div {
        background-color: #161b22;
        color: white;
    }

    .css-1x8cf1d {
        color: #c9d1d9 !important;
    }

    h1, h2, h3, h4 {
        color: #f0f6fc;
    }

    .stButton>button {
        background-color: #238636;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 16px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #2ea043;
    }

    .stSelectbox>div {
        background-color: #161b22;
    }

    .stSidebar {
        background-color: #161b22 !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.header("🎛️ Filters")
    all_genres = sorted({g for sub in movies['genre_list'] for g in sub})
    selected_genres = st.multiselect("Filter by Genres", all_genres)
    st.markdown("---")
    st.write("👋 Try entering a movie title or use Surprise Me!")

# --- Main UI ---
st.title(" 🎬 Movieflix - Movie Recommender")

user_id = st.number_input("User ID (for CF simulation)", min_value=1, max_value=1000, value=1, step=1)
movie_query = st.text_input("Search a movie you like:")
suggestions = [title for title in movies['title'].values if movie_query.lower() in title.lower()][:10]

selected_movie = st.selectbox("Select a movie from suggestions:", suggestions) if suggestions else None

if st.button("🎲 Surprise Me!"):
    selected_movie = np.random.choice(movies['title'].values)
    st.success(f"How about watching **{selected_movie}**?")

if selected_movie:
    st.subheader(f"Top recommendations based on **{selected_movie}**")
    recommendations = hybrid_recommendation(user_id, selected_movie, top_n=10, content_weight=0.5, cf_weight=0.5, selected_genres=selected_genres)
    if recommendations:
         num_cols = 5
    cols = st.columns(num_cols)

    for idx, rec in enumerate(recommendations):
        with cols[idx % num_cols]:
            st.image(rec['poster_url'] if rec['poster_url'] else 'https://via.placeholder.com/150', width=150)
            st.markdown(f"<div class='movie-title'>{rec['title']}</div>", unsafe_allow_html=True)
            st.markdown(get_stars(rec['rating']))
            st.markdown(f"<a class='trailer-link' href='{get_youtube_link(rec['title'])}' target='_blank'>🎥 Trailer</a>", unsafe_allow_html=True)


st.header("🔥 Top 10 Movies Overall")
top_10 = get_top_10_movies_overall(movies)
cols = st.columns(5)
for idx, movie in enumerate(top_10):
    with cols[idx % 5]:
        if movie['poster_url']:
            st.image(movie['poster_url'], use_column_width=True)
        st.markdown(f"<div class='movie-title'>{movie['title']}</div>", unsafe_allow_html=True)

st.markdown("---")
