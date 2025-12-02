# vectorizer.py
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def build_vectorizer(max_features=6000):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )

def save_vectorizer(vectorizer, path="tfidf_vectorizer.pkl"):
    joblib.dump(vectorizer, path)

def load_vectorizer(path="tfidf_vectorizer.pkl"):
    return joblib.load(path)
