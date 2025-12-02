# train_gda.py
import numpy as np
import joblib

def compute_gda_params(X, y, eps=1e-6):
    X0 = X[y == 0]
    X1 = X[y == 1]

    pi0 = len(X0) / len(X)
    pi1 = len(X1) / len(X)

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    Sigma = ((X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1)) / len(X)
    Sigma += eps * np.eye(Sigma.shape[0])  # regularization
    Sigma_inv = np.linalg.inv(Sigma)

    params = {
        "pi0": pi0, "pi1": pi1,
        "mu0": mu0, "mu1": mu1,
        "Sigma": Sigma,
        "Sigma_inv": Sigma_inv
    }
    joblib.dump(params, "gda_model.pkl")
    return params


def gda_predict(X, params):
    mu0, mu1 = params["mu0"], params["mu1"]
    pi0, pi1 = params["pi0"], params["pi1"]
    Sigma_inv = params["Sigma_inv"]

    def score(x, mu, pi):
        return x @ Sigma_inv @ mu - 0.5 * mu.T @ Sigma_inv @ mu + np.log(pi)

    s0 = np.apply_along_axis(lambda x: score(x, mu0, pi0), 1, X)
    s1 = np.apply_along_axis(lambda x: score(x, mu1, pi1), 1, X)
    return (s1 > s0).astype(int), s1
