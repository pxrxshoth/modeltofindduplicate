# 🔍 Duplicate Question Detection using ML

This project focuses on detecting duplicate questions using the **Quora Question Pairs dataset**. Various classification algorithms were trained and evaluated to identify the most effective model for this natural language processing (NLP) task.

---

## 📌 Objective

> **To build a robust machine learning system that identifies whether two questions are semantically similar (duplicates) using the Quora Question Pair dataset.**

---

## 📊 Models Trained

The following classification models were implemented and compared:

- Extra Trees Classifier
- XGBoost
- Decision Tree Classifier
- Gradient Boosting Classifier
- Stochastic Gradient Boosting
- CatBoost
- AdaBoost Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

Each model was evaluated based on metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.

---

## 🏆 Outcome

> Trained and evaluated various classification models and highlighted the best-performing model for identifying duplicate questions using the Quora Question Pair dataset.

---

## 📁 Dataset

- Dataset used: [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)
- Format: CSV containing question pairs and labels (`is_duplicate`)

---

## 🧪 Features & Techniques

- Text preprocessing and tokenization
- Feature engineering: word match share, TF-IDF vectors, and sentence embeddings
- Handling imbalanced classes
- Hyperparameter tuning
- Evaluation using confusion matrix and cross-validation

---

## 📦 Tech Stack

- Python
- Scikit-learn
- XGBoost, CatBoost, LightGBM
- Pandas, NumPy, Matplotlib, Seaborn
- Jupyter Notebook

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/pxrxshoth/modeltofindduplicate.git
   cd modeltofindduplicate
