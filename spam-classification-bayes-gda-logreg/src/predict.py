# predict.py
import joblib
from clean_text import clean_text

def load_pipeline(model_path="final_model.pkl", vect_path="tfidf_vectorizer.pkl"):
    model = joblib.load(model_path)
    vect = joblib.load(vect_path)
    return model, vect

def predict_message(text, model, vect):
    cleaned = clean_text(text)
    vec = vect.transform([cleaned])
    prob = model.predict_proba(vec)[0,1]
    label = "spam" if prob >= 0.5 else "ham"
    return label, prob

# Example
if __name__ == "__main__":
    model, vect = load_pipeline()
    msg = "Congratulations! You won a free gift voucher!"
    label, prob = predict_message(msg, model, vect)
    print(label, prob)
