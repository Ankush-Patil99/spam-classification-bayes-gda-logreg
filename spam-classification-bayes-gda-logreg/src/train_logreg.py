# train_logreg.py
import joblib
from sklearn.linear_model import LogisticRegression

def train_logreg(X_train, y_train, save_path="logreg_model.pkl"):
    model = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model
