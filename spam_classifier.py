"""
Email Spam Classifier Using Machine Learning
============================================
Dataset : SMS Spam Collection (UCI / Kaggle)
          https://www.kaggle.com/uciml/sms-spam-collection-dataset

Steps
-----
1. Load & explore dataset
2. Text preprocessing (lower-case, punctuation removal, stop-words, stemming)
3. Feature extraction with TF-IDF
4. Train / evaluate multiple classifiers
5. Compare results and persist the best model
"""

import os
import re
import string
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# NLTK resources
# ---------------------------------------------------------------------------
for resource in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{resource}" if resource == "stopwords" else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------

DATASET_PATH = "spam.csv"  # place the downloaded CSV here


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the SMS Spam Collection dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from https://www.kaggle.com/uciml/sms-spam-collection-dataset "
            "and place spam.csv in the project directory."
        )

    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart of ham vs spam counts."""
    counts = df["label"].value_counts()
    labels = ["Ham (Not Spam)", "Spam"]
    colors = ["#4CAF50", "#F44336"]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=counts.values, palette=colors)
    plt.title("Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.show()
    print(f"\nClass distribution:\n{counts.rename({0: 'Ham', 1: 'Spam'})}\n")


# ---------------------------------------------------------------------------
# 3. Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Lowercase → remove punctuation/digits → tokenise → remove stop-words → stem."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_message"] = df["message"].apply(preprocess_text)
    return df


# ---------------------------------------------------------------------------
# 4. Feature extraction
# ---------------------------------------------------------------------------

def build_tfidf_features(
    X_train: pd.Series,
    X_test: pd.Series,
    max_features: int = 5000,
):
    """Fit TF-IDF on training data and transform both splits."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer


# ---------------------------------------------------------------------------
# 5. Model training & evaluation
# ---------------------------------------------------------------------------

MODELS = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    "Linear SVM": LinearSVC(C=1.0, max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """Train a model and return a metrics dictionary."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
    }

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
    )
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png", dpi=150)
    plt.show()

    return metrics


def compare_models(results: list[dict]) -> None:
    """Print and plot a comparison of all model metrics."""
    results_df = pd.DataFrame(results).set_index("Model")
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string())

    results_df.plot(kind="bar", figsize=(10, 5), ylim=(0.9, 1.0), rot=15)
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# 6. Persist best model
# ---------------------------------------------------------------------------

def save_model(model, vectorizer, model_path: str = "spam_model.pkl", vec_path: str = "tfidf_vectorizer.pkl") -> None:
    """Serialize the trained model and vectorizer with pickle."""
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\nModel saved → {model_path}")
    print(f"Vectorizer saved → {vec_path}")


def load_model(model_path: str = "spam_model.pkl", vec_path: str = "tfidf_vectorizer.pkl"):
    """Load persisted model and vectorizer."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# ---------------------------------------------------------------------------
# 7. Prediction helper
# ---------------------------------------------------------------------------

def predict_message(message: str, model, vectorizer) -> str:
    """Classify a single raw message string."""
    cleaned = preprocess_text(message)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    return "SPAM" if prediction == 1 else "HAM (Not Spam)"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Email Spam Classifier — ML Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_dataset()
    print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.head())

    # 2. EDA
    plot_class_distribution(df)

    # 3. Preprocess
    df = preprocess_dataframe(df)

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    print(f"\nTrain size: {len(X_train)}   Test size: {len(X_test)}")

    # 5. TF-IDF features
    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(X_train, X_test)

    # 6. Train & evaluate all models
    results = []
    trained_models = {}
    for name, model in MODELS.items():
        metrics = evaluate_model(name, model, X_train_tfidf, X_test_tfidf, y_train, y_test)
        results.append(metrics)
        trained_models[name] = model

    compare_models(results)

    # 7. Pick best model by F1-Score and save
    best = max(results, key=lambda r: r["F1-Score"])
    print(f"\nBest model: {best['Model']}  (F1 = {best['F1-Score']:.4f})")
    save_model(trained_models[best["Model"]], vectorizer)

    # 8. Quick demo
    demo_messages = [
        "Congratulations! You've won a FREE iPhone. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your bank account has been suspended. Verify immediately.",
        "The project report is due on Friday. Please review before sending.",
    ]
    print("\n" + "=" * 60)
    print("  Demo Predictions")
    print("=" * 60)
    for msg in demo_messages:
        result = predict_message(msg, trained_models[best["Model"]], vectorizer)
        print(f"[{result}]  →  {msg[:70]}")


if __name__ == "__main__":
    main()
