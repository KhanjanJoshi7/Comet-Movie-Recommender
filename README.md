# 🎬 Comet Movie Recommender

A hybrid machine learning-powered movie recommendation system that predicts whether a user will like a movie and generates personalized recommendations — deployed as an interactive Streamlit web app.

🔗 **Live App:** [comet-movie-recommender.streamlit.app](https://comet-movie-recommender-bzuudjzkqgnp5mu7m6njxc.streamlit.app/)

---

## 📌 Project Overview

The Comet Movie Recommender tackles the "paradox of choice" on modern streaming platforms by combining two complementary recommendation strategies:

- **Content-Based Filtering** — analyzes movie metadata (genres, plot, cast, director) using TF-IDF vectorization and cosine similarity
- **Collaborative Filtering** — learns from user rating patterns using Singular Value Decomposition (SVD)

Together, these approaches overcome cold-start and data sparsity problems to deliver accurate, diverse, and personalized recommendations.

---

## 🗂️ Repository Structure

```
comet-movie-recommender/
│
├── streamlit_app.py              # Main Streamlit web application
├── movie_like_predictor.pkl      # Trained XGBoost model (like/dislike predictor)
├── genre_encoder.pkl             # MultiLabelBinarizer for genre encoding
│
├── notebooks/
│   ├── AML_EDA.ipynb             # Exploratory Data Analysis
│   ├── AML_Final_XGBoost_Accuracy.ipynb  # XGBoost model training & evaluation
│   └── movie-recommender.ipynb   # Hybrid recommender system (SVD + content-based)
│
├── Movie-Recommender.pdf         # Full project technical report
└── README.md
```

---

## 📊 Dataset

Data is sourced from **IMDb** and merged with a **MovieLens**-style user ratings dataset.

📁 [Download Dataset from Google Drive](https://drive.google.com/drive/folders/17XF0PTbLMd8xRfC74efT3JKSLg50n_9k?usp=sharing)

| File | Description |
|------|-------------|
| `movies.csv` | Movie metadata: title, genres, year |
| `ratings.csv` | User-movie ratings (userId, movieId, rating) |
| `links.csv` | TMDB/IMDb ID mappings |
| `IMDB Dataset.csv` | Movie reviews with sentiment labels |

---

## 🤖 Models

### 1. Random Forest Classifier — 92.44% Accuracy
Predicts whether a user will like a movie based on `user_avg_rating`, `user_activity`, `year`, and one-hot encoded genres. Trained with 100 estimators and max depth of 12.

### 2. Logistic Regression (Sentiment Analysis) — 85.08% Accuracy
Classifies IMDB movie reviews as positive or negative using TF-IDF vectorization (top 1,000 features) on 20,000 reviews.

### 3. XGBoost Classifier — 92.24% Accuracy ⭐ (Deployed)
The primary like/dislike predictor. Trained on 30,000 records using behavioral features (`user_avg_rating`, `user_activity`) and genre encodings. F1-score of 0.92 for both classes.

### 4. Hybrid Recommender (SVD + Cosine Similarity)
Combines matrix factorization (via the `Surprise` library) with TF-IDF content similarity for end-to-end movie recommendations.

---

## 🧪 Model Pipeline

```
Movie Data + Review Data
        ↓
NLP Preprocessing (Stop words, POS Tagging)
        ↓
Word2Vec → Auto-labeling (Pos/Neg)
        ↓
Feature Engineering (Genres, Year, User Behavior)
        ↓
Train / Validate / Test Split (80/20)
        ↓
XGBoost | Random Forest | Logistic Regression
        ↓
Streamlit Deployment
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit pandas scikit-learn xgboost joblib scikit-surprise
```

### Run the App Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/comet-movie-recommender.git
   cd comet-movie-recommender
   ```

2. Download the dataset from the [Google Drive link](https://drive.google.com/drive/folders/17XF0PTbLMd8xRfC74efT3JKSLg50n_9k?usp=sharing) and place the CSV files in the project root.

3. Update the model paths in `streamlit_app.py` to use relative paths:
   ```python
   model = joblib.load("movie_like_predictor.pkl")
   mlb = joblib.load("genre_encoder.pkl")
   ```

4. Launch the app:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 🖥️ App Features

- 🎥 Select a movie and apply filters (genre, year, minimum rating)
- 🤝 Get collaborative filtering-based recommendations
- 😊 View sentiment analysis on mock reviews (positive/negative)
- 👍 Like/dislike prediction from XGBoost and Random Forest
- 🎨 Custom-styled UI with themed background and layout

---

## 📈 Key EDA Insights

- **Most Rated:** The Shawshank Redemption (1994), Forrest Gump (1994), Pulp Fiction (1994)
- **Highest Rated (min 500 ratings):** Planet Earth II (2016), Planet Earth (2006), Band of Brothers (2001)
- **Best Genre by Average Rating:** Film-Noir > War > Crime > Documentary
- **User Behavior:** Most users rate only 1–10 movies (heavily right-skewed distribution)
- **Rating Activity Peak:** 1999–2001, with a secondary bump around 2015–2016 and 2020

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.x |
| ML / Modeling | scikit-learn, XGBoost, Surprise (SVD) |
| NLP | TF-IDF, Word2Vec, NLTK |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Deployment | Streamlit, Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 📄 License

This project was developed for academic purposes at the University of Texas at Dallas. All datasets are used under their respective open-access terms (MovieLens, TMDB, IMDb).

---

## 🙏 Acknowledgements

- Professor Ziyi Cao for guidance throughout the project
- [MovieLens](https://grouplens.org/datasets/movielens/) for the ratings dataset
- [IMDb](https://www.imdb.com/) for movie metadata
