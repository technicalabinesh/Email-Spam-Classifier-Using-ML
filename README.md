# 📧 Email Spam Classifier Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Objective

Build a machine-learning pipeline that classifies emails (or SMS messages) as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and multiple ML algorithms.

---

## 🗂️ Project Structure

```
Email-Spam-Classifier-Using-ML/
│
├── spam_classifier.py        # Full ML pipeline (main script)
├── spam.csv                  # Dataset (download separately — see below)
├── spam_model.pkl            # Saved best model (generated after running)
├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer (generated after running)
│
├── class_distribution.png   # EDA chart (generated after running)
├── model_comparison.png      # Model comparison chart (generated after running)
└── README.md
```

---

## 🧰 Tech Stack

| Tool / Library | Purpose |
|---|---|
| Python 3.8+ | Core language |
| Pandas / NumPy | Data manipulation |
| NLTK | Text preprocessing (stop-words, stemming) |
| Scikit-learn | TF-IDF, ML models, evaluation |
| Matplotlib / Seaborn | Visualisation |
| Pickle | Model persistence |

---

## 📁 Dataset

Uses the **SMS Spam Collection** dataset (UCI / Kaggle).

1. Download `spam.csv` from:  
   👉 <https://www.kaggle.com/uciml/sms-spam-collection-dataset>

2. Place the file in the **project root directory** (same folder as `spam_classifier.py`).

---

## 🚀 Getting Started

### 1 — Install dependencies

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

### 2 — Run the classifier

```bash
python spam_classifier.py
```

The script will:

- Load and explore the dataset
- Pre-process text (lower-case, punctuation removal, stop-word filtering, stemming)
- Extract TF-IDF features (unigrams + bigrams, top 5 000 features)
- Train and evaluate **four** classifiers
- Save the best-performing model to `spam_model.pkl`
- Demonstrate predictions on sample messages

---

## 🤖 Models Compared

| Model | Notes |
|---|---|
| Naive Bayes | Fast, strong baseline for text |
| Logistic Regression | Interpretable, well-calibrated |
| Linear SVM | High accuracy on sparse TF-IDF vectors |
| Random Forest | Ensemble, robust to overfitting |

---

## 🧪 Pipeline Overview

```
Raw text
   │
   ▼
Lower-case → Remove digits/punctuation → Tokenise
   │
   ▼
Remove stop-words → Porter Stemming
   │
   ▼
TF-IDF Vectoriser (unigrams + bigrams, max 5 000 features)
   │
   ▼
Train / Evaluate (Accuracy · Precision · Recall · F1)
   │
   ▼
Best model serialised with pickle
```

---

## 📊 Sample Output

```
Model             Accuracy  Precision  Recall   F1-Score
Naive Bayes       0.9767    0.9744     0.8929   0.9320
Logistic Reg.     0.9838    0.9855     0.9107   0.9466
Linear SVM        0.9847    0.9861     0.9152   0.9493
Random Forest     0.9776    0.9875     0.8750   0.9279
```

*(Actual numbers vary slightly by random seed and dataset version.)*

---

## 🔮 Predicting New Messages

```python
from spam_classifier import load_model, predict_message

model, vectorizer = load_model()
print(predict_message("Congratulations! You've won a FREE iPhone!", model, vectorizer))
# → SPAM
print(predict_message("See you at the meeting tomorrow.", model, vectorizer))
# → HAM (Not Spam)
```

---

## 🧩 Optional Enhancements

- [ ] Word2Vec / BERT embeddings for richer text features
- [ ] Hyperparameter tuning with `GridSearchCV`
- [ ] Flask / Streamlit web app for live classification
- [ ] Email header analysis (sender, subject) as additional features
- [ ] Deployment on AWS / GCP / Heroku

---

## 🎯 Conclusion

This project demonstrates a complete, end-to-end spam classification pipeline — from raw text to a persisted model ready for deployment. It can be extended into a web service or integrated directly into email clients for real-world spam filtering.

---

## 📄 License

This project is licensed under the **MIT License**.
