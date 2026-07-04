# Fluency-Analysis-of-Students-Viva-from-Voice-Signals
An Explainable Framework for Automated Speech Fluency Evaluation using Spectral Features and Machine Learning

## 📌 Overview

This repository presents an explainable machine learning framework for automated speech fluency evaluation. The system classifies speech recordings into **Fluent** and **Non-Fluent** categories using handcrafted acoustic features and statistical machine learning algorithms.

The proposed framework extracts multiple spectral features from speech, performs feature engineering and dimensionality reduction, and applies several machine learning models to identify speech fluency accurately. Explainable AI techniques such as **LIME** and **SHAP** are incorporated to interpret model predictions.

This work has been published as an IEEE conference paper.

---

## 🚀 Features

- Audio preprocessing
- Noise reduction
- Silence removal
- Feature normalization
- MFCC feature extraction
- LPC feature extraction
- DCT feature extraction
- Feature fusion
- Correlation-based feature filtering
- Sequential Feature Selection (SFS)
- Principal Component Analysis (PCA)
- Machine Learning classification
- Explainable AI using SHAP
- Explainable AI using LIME

---

## 📂 Dataset

The dataset consists of manually labeled speech fluency samples collected from student viva responses.

Classes:

- Fluent
- Non-Fluent

The processed feature dataset is available in:

```text
dataset/fluency_dataset_auto_labeled.csv
```

---

## 🛠️ Machine Learning Models

The following classifiers were implemented and evaluated:

- Random Forest
- Decision Tree
- Support Vector Machine (SVM)
- Logistic Regression
- k-Nearest Neighbors (kNN)
- Naïve Bayes
- XGBoost
- Multi-Layer Perceptron (MLP)
- Stacking Ensemble

---

## 🔬 Methodology

The complete workflow consists of:

1. Audio Collection
2. Audio Preprocessing
3. Noise Removal
4. Silence Trimming
5. Feature Extraction
   - MFCC
   - LPC
   - DCT
6. Feature Fusion
7. Correlation Analysis
8. Sequential Feature Selection
9. Principal Component Analysis
10. Model Training
11. Hyperparameter Tuning
12. Model Evaluation
13. Model Explainability using SHAP and LIME

---
