

# 📧 SMS Spam Classification — Naive Bayes, GDA, Logistic Regression
A complete end-to-end SMS spam detection project combining classical
machine-learning models with probability-driven approaches. This project
is clean, professional, and GitHub-ready.

## 🚀 PROJECT HIGHLIGHTS
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

<details>
<summary><h2>📦 Full Project Folder Structure (Click to Expand)</h2></summary>

<br>

## 📁 models/
- 📄 [final_model.pkl](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/models/final_model.pkl)
- 📄 [tfidf_vectorizer.pkl](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/models/tfidf_vectorizer.pkl)

---

## 📁 notebooks/
- 📓 [spam_classification.ipynb](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/notebooks/spam_classification.ipynb)

---

## 📁 plots/
- 🟦 [feature_importance_logreg.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/plots/feature_importance_logreg.png)
- 🟥 [final_model_comparison.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/plots/final_model_comparison.png)
- 🟪 [multiclassification_examples.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/plots/multiclassification_examples.png)
- 🟩 [top_spam_tokens_MNB.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/plots/top_spam_tokens_MNB.png)

---

## 📁 results/
- 📊 [calibration_plot.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/results/calibration_plot.png)
- 🟦 [confusion_matrix_logreg.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/results/confusion_matrix_logreg.png)
- 🟨 [confusion_matrix_mnb.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/results/confusion_matrix_mnb.png)
- 📑 [model_comparison_results.csv](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/results/model_comparison_results.csv)
- 📈 [precision_recall_curve.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/results/precision_recall_curve.png)
- 📉 [ROC_curve.png](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/results/ROC_curve.png)

---

## 📁 src/
- 🧹 [clean_text.py](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/src/clean_text.py)
- 🤖 [predict.py](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/src/predict.py)
- 🔷 [train_gda.py](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/src/train_gda.py)
- 🔴 [train_logreg.py](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/src/train_logreg.py)
- 🟡 [train_nb.py](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/src/train_nb.py)
- 🧩 [vectorizer.py](https://github.com/Ankush-Patil99/spam-classification-bayes-gda-logreg/blob/main/spam-classification-bayes-gda-logreg/src/vectorizer.py)

</details>
  

## 🧹PREPROCESSING STEPS
1. Lowercasing  
2. URL removal  
3. Number removal  
4. Punctuation cleaning  
5. Stopword removal  
6. Token filtering  
7. TF-IDF vectorization (uni + bi-grams)


## 🧠 MODELS TRAINED
### 💛 Naive Bayes Family
- MultinomialNB → strong for sparse text  
- BernoulliNB → binary features  
- GaussianNB → baseline comparison  
### 🔵 Gaussian Discriminant Analysis — Manual Implementation
- Mean vectors  
- Shared covariance matrix  
- Priors  
- Discriminant score function  
### ❤️ Logistic Regression
- Highest performance  
- Clean probability estimates  
- Great interpretability  


## 📈 EVALUATION METRICS
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- PR-AUC  
- Calibration curve  
- Confusion matrices  

## 🖼 VISUALIZATIONS INCLUDED
- Confusion matrices (MNB, LogReg)  
- ROC curve  
- Precision-Recall curve  
- Calibration plot  
- Logistic Regression feature importance plot  
- Top spam tokens (Multinomial NB)  
- Misclassified samples analysis  


## 📘 PREDICTION EXAMPLE
label, prob = predict_message("Congratulations! You won a free gift!")  
Output: spam (prob ≈ 0.98)


## 🏁 SUMMARY
This project is a full, professional workflow:
- Clean modular code  
- Strong evaluation  
- Interpretable models  
- Real-world ready prediction pipeline  
- Perfect for GitHub portfolio & interviews  

## 👨‍💻 AUTHOR

Well-structured by a dedicated ML engineer aiming to master classical + probabilistic learning.
