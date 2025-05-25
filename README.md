# Hybrid Movie Recommendation System

A content + collaborative filtering-based movie recommender system with a beautiful **Streamlit** web interface. Built using **NLP (TF-IDF)**, **KNN**, and a simulated **collaborative filtering** model, this hybrid system provides accurate and personalized movie recommendations.  

<img src="https://user-images.githubusercontent.com/123456789/your_demo_gif.gif" width="700"/>

---

##  Features

- **Hybrid Recommendation**: Combines content similarity and simulated user preferences.
- **Movie Posters & Trailers**: Integrates with TMDb and YouTube.
- **Interactive UI**: Built with Streamlit, styled with HTML/CSS.
- **Genre Filters**: Customize recommendations by genre.
- **"Surprise Me"** Feature: Get random movie suggestions.
- **Deployed Web App**: [View Demo](https://your-deployment-url)

---

## How it Works

### 1. **Content-Based Filtering**
- Extracts keywords from:
  - Movie overview
  - Genres
  - Cast
  - Crew (Director)
  - Keywords
- Uses **TF-IDF Vectorizer** + **KNN (cosine similarity)**

### 2. **Collaborative Filtering (Simulated)**
- Predicts ratings using a seeded random function (replaceable with real CF model like SVD or ALS).

### 3. **Hybrid Score**
- Weighted average of content similarity and CF rating:
  
  \[
  \text{Hybrid Score} = w_1 \times \text{Content Score} + w_2 \times \text{CF Score}
  \]

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Yashaswikathota/Movie-Recommendation-System.git
cd Movie-Recommendation-System
