# Spam Mail Prediction using Machine Learning

A complete, comparison-driven machine learning pipeline that classifies messages as **spam** or **ham** (legitimate), built with scikit-learn. Rather than stopping at a single model and a single accuracy number, this project compares multiple algorithms, investigates a data-leakage issue, and analyzes the precision-recall trade-off in detail.

## Project Overview

- **Task:** Binary text classification (spam vs. ham)
- **Approach:** TF-IDF feature extraction + comparison of 5 model configurations across 3 algorithm families
- **Dataset:** ~11,000 labeled complex mix of messages, ~79% ham / 21% spam
- **Best model:** Support Vector Machine (SVM), achieving a balance between 97.9% test accuracy with 0.95 F1-score on the spam class

## Why this project is more than "just" a classifier

1. **Found and fixed a data leakage issue** — ~25% of the dataset was exact duplicates, which could otherwise appear in both train and test sets and inflate reported accuracy.
2. **Diagnosed class imbalance's real impact** — accuracy alone looked fine (95.6%), but the confusion matrix revealed the model was missing ~19% of real spam messages. Precision/recall/F1 told the true story.
3. **Compared 5 model configurations** across 3 different algorithm families, not just one.
4. **Tested decision threshold tuning** — including a hypothesis that was disproven by the data, and reported that honestly rather than cherry-picking results.
5. **Reasoned about real-world error costs** — explicitly weighing whether a false positive (real message flagged as spam) or false negative (spam reaching the inbox) is worse, and how that should guide model choice.

## Dataset

The dataset contains two columns:
| Column | Description |
|---|---|
| `Category` | Label: `spam` or `ham` |
| `Message` | Raw message text |

**Class distribution:**
- Ham: ~8,825 messages (~79%)
- Spam: ~2,352 messages (~21%)

## Pipeline

```
Raw CSV
   │
   ▼
Handle null values
   │
   ▼
Remove duplicate rows  ──────►  (prevents data leakage across train/test split)
   │
   ▼
Label encoding (spam=0, ham=1)
   │
   ▼
Train/test split (80/20)
   │
   ▼
TF-IDF vectorization  ─────────►  fit_transform on train, transform only on test
   │
   ▼
Train & evaluate 5 model configurations
   │
   ▼
Compare via accuracy, precision, recall, F1, confusion matrices
   │
   ▼
Threshold tuning on probability outputs
   │
   ▼
Predictive system for new, unseen messages
```

## Models Compared

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|---|---|---|---|---|
| Naive Bayes | 90.4% | 0.99 | 0.56 | 0.72 |
| Logistic Regression (default) | 95.6% | 0.99 | 0.81 | 0.89 |
| Logistic Regression (balanced) | 97.0% | 0.92 | 0.94 | 0.93 |
| **SVM (default)** | **97.9%** | **0.98** | **0.93** | **0.95** |
| SVM (balanced) | 98.0% | 0.97 | 0.94 | 0.95 |

**Final model:** SVM (default) — chosen for the strongest overall balance of precision and recall, minimizing both false alarms on real messages and missed spam.

## Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn (TfidfVectorizer, LogisticRegression, MultinomialNB, LinearSVC)
- matplotlib, seaborn (visualizations)
- Jupyter Notebook / Google Colab
