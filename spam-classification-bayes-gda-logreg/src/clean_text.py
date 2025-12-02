# clean_text.py
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 1]
    return " ".join(tokens)
