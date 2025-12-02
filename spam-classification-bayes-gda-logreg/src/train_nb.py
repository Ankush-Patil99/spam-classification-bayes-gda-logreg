# train_nb.py
import joblib
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

def train_multinomial_nb(X_train, y_train, save_path="mnb_model.pkl"):
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model

def train_bernoulli_nb(X_train_bin, y_train, save_path="bnb_model.pkl"):
    model = BernoulliNB(alpha=0.1)
    model.fit(X_train_bin, y_train)
    joblib.dump(model, save_path)
    return model

def train_gaussian_nb(X_train_dense, y_train, save_path="gnb_model.pkl"):
    model = GaussianNB()
    model.fit(X_train_dense, y_train)
    joblib.dump(model, save_path)
    return model
