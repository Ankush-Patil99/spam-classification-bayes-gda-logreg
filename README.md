

# 📧 SMS Spam Classification — Naive Bayes, GDA, Logistic Regression
A complete end-to-end SMS spam detection project combining classical
machine-learning models with probability-driven approaches. This project
is clean, professional, and GitHub-ready.

## 🚀 PROJECT HIGHLIGHTS:-

✔ Multinomial Naive Bayes  
✔ Bernoulli Naive Bayes  
✔ Gaussian Naive Bayes  
✔ Gaussian Discriminant Analysis (GDA) — implemented from scratch  
✔ Logistic Regression (benchmark model)  
✔ TF-IDF text vectorization  
✔ Strong evaluation: ROC, PR curve, calibration, confusion matrices  
✔ Feature importance, token importance, misclassification analysis  
✔ Modular src/ folder for training + prediction scripts  
✔ Saved models for real-world inference  
---
## 📑 Table of Contents
- [🚀 Project Highlights](#-project-highlights)
- [📦 Folder Structure](#-full-project-folder-structure-click-to-expand)
- [🧹 Preprocessing Steps](#-preprocessing-steps)
- [🧠 Models Trained](#-models-trained)
  - [1️⃣ Naive Bayes Family](#1-naive-bayes-family)
  - [2️⃣ Gaussian Discriminant Analysis (GDA)](#2-gaussian-discriminant-analysis-gda--implemented-from-scratch)
  - [3️⃣ Logistic Regression](#3-logistic-regression-best-model)
- [📈 Evaluation Metrics](#-evaluation-metrics)
- [🖼 Visualizations Included](#-visualizations-included)
- [📘 Prediction Example](#-prediction-example)
- [🏁 Summary](#-summary)
- [🧰 Tech Stack](#-tech-stack)
- [👤 Author](#-author)
---

<details>
<summary><h2>📦 Full Project Folder Structure (Click to Expand)</h2></summary>

<br>

## 📁 models/
-  [final_model.pkl](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/models/final_model.pkl)
-  [tfidf_vectorizer.pkl](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/models/tfidf_vectorizer.pkl)

---

## 📁 notebooks/
- 📓 [spam_classification.ipynb](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/notebooks/spam_classification.ipynb)

---

## 📁 plots/
-  [feature_importance_logreg.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/plots/feature_importance_logreg.png)
-  [final_model_comparison.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/plots/final_model_comparison.png)
-  [multiclassification_examples.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/plots/multiclassification_examples.png)
-  [top_spam_tokens_MNB.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/plots/top_spam_tokens_MNB.png)

---

## 📁 results/
-  [calibration_plot.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/results/calibration_plot.png)
-  [confusion_matrix_logreg.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/results/confusion_matrix_logreg.png)
-  [confusion_matrix_mnb.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/results/confusion_matrix_mnb.png)
-  [model_comparison_results.csv](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/results/model_comparison_results.csv)
-  [precision_recall_curve.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/results/precision_recall_curve.png)
-  [ROC_curve.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/results/ROC_curve.png)

---

## 📁 src/
-  [clean_text.py](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/src/clean_text.py)
-  [predict.py](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/src/predict.py)
-  [train_gda.py](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/src/train_gda.py)
-  [train_logreg.py](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/src/train_logreg.py)
-  [train_nb.py](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/src/train_nb.py)
-  [vectorizer.py](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/src/vectorizer.py)

</details>
  

## 🧹 Preprocessing Steps
The SMS messages go through a complete cleaning + vectorization pipeline:

1. **Lowercasing** – normalize text  
2. **URL removal** – remove hyperlinks  
3. **Number removal** – remove digit-heavy tokens  
4. **Punctuation cleaning** – strip special characters  
5. **Stopword removal** – remove common non-informative words  
6. **Token filtering** – remove very short or meaningless tokens  
7. **TF-IDF Vectorization**  
   - Uni-grams + Bi-grams  
   - `max_features = 6000`  
   - `min_df = 2`  
   - Produces a high-quality sparse representation of text  


---

## 🧠 Models Trained

### 1. Naive Bayes Family
- **MultinomialNB** → Best for TF-IDF sparse matrices  
- **BernoulliNB** → Binary text features  
- **GaussianNB** → Baseline comparison  

---

###  2. Gaussian Discriminant Analysis (GDA) — *Implemented From Scratch*
GDA was implemented manually using the original mathematical formulation:

- Compute **mean vectors** μ₀, μ₁  
- Compute **shared covariance matrix** Σ  
- Add small regularization (εI)  
- Compute **priors** π₀, π₁  
- Implement discriminant function: **δ(x) = xᵀ Σ⁻¹ μ - ½ μᵀ Σ⁻¹ μ + log(π)**

GDA surprisingly performs well even on dense TF-IDF vectors.

---
###  3. Logistic Regression (Best Model)
- **Highest precision, recall and F1-score**  
- Best **calibrated probability outputs**  
- Highly interpretable coefficients  
- Lightweight and robust for text classification 

## 📈 Evaluation Metrics  
The models were evaluated using multiple classification and probability-quality metrics:

- **Accuracy** – overall correctness  
- **Precision** – how many predicted spam messages were actually spam  
- **Recall** – ability to detect spam messages  
- **F1 Score** – balanced metric between precision and recall  
- **ROC–AUC** – ability to separate classes across thresholds  
- **PR–AUC** – especially useful for imbalanced spam datasets  
- **Calibration Curve** – probability correctness (LogReg performs best)  
- **Confusion Matrices** – detailed class-wise error breakdown  

---

## 🖼 Visualizations Included  
The project includes rich visual diagnostic plots for performance and interpretability:

- **Confusion Matrices** → MultinomialNB & Logistic Regression  
- **ROC Curve** → classifier separability  
- **Precision–Recall Curve** → performance on imbalanced data  
- **Calibration Plot** → probability quality comparison  
- **Feature Importance Plot (LogReg)** → top predictive words  
- **Top Spam Tokens (MNB)** → strongest spam indicators  
- **Misclassified Samples Analysis** → understand incorrect predictions  

---  
## 📘 PREDICTION EXAMPLE

You can test any SMS message using the `predict.py` script:
```python
from src.predict import load_pipeline, predict_message
model, vect = load_pipeline()
label, prob = predict_message("Congratulations! You won a free gift!")
print(label, prob)
```
Output:
```
spam  (probability ≈ 0.98)
```
This clean interface makes the model easy to integrate with APIs, mobile apps, or automation pipelines.

---
## 🏁 SUMMARY


This project delivers a complete, production-ready classical NLP pipeline:
- Modular codebase (src/) for easy reuse  
- Comprehensive evaluation using ROC, PR curve, calibration, confusion matrices  
- Interpretable models with token importance, feature importance, and misclassification analysis  
- Deployment-ready design with saved models & prediction script  
- Excellent GitHub & resume project, showcasing classical ML mastery and mathematical depth (GDA from scratch)  
---
## 🧰 Tech Stack

**Languages**
- 🟦 Python 3.10+

**Core Libraries**
- 🧮 NumPy  
- 📊 Pandas  
- 🔤 Scikit-Learn  
- 🧰 SciPy  

**NLP & Text Processing**
- 🔡 NLTK  
- 🧾 TF-IDF Vectorizer  

**Visualization**
- 📈 Matplotlib  
- 🎨 Seaborn  

**Model Persistence**
- 💾 joblib  

**Environment**
- 🧪 Jupyter Notebook  
- 🗂 GitHub Repository Structure  

---

## 👤 Author
**Ankush Patil**  
Machine Learning & NLP Engineer  
📧 **Email**: ankpatil1203@gmail.com  
💼 **LinkedIn**: www.linkedin.com/in/ankush-patil-48989739a  
🌐 **GitHub**: https://github.com/Ankush-Patil99  

Well-structured by a dedicated ML engineer aiming to master classical + probabilistic learning.
