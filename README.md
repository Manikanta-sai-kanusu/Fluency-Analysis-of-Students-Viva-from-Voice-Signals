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


# 🎙️ Fluency Analysis of Students Viva from Voice Signals

<div align="center">

<img src="assets/banner.png" alt="Fluency Analysis Banner" width="900"/>

### 🧠 Explainable Machine Learning for Automated Speech Fluency Assessment

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Audio-Librosa-FF6F00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-8E44AD?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/IEEE-Published-00629B?style=for-the-badge&logo=ieee"/>
</p>

<p>
  <a href="#-overview">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-workflow">Workflow</a> •
  <a href="#-results">Results</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-publication">Publication</a>
</p>

</div>

---

## 🚀 Overview

**Fluency Analysis of Students Viva from Voice Signals** is a machine learning project that automatically analyzes a student's viva voice recording and classifies the speech as:

<div align="center">

|           🟢 Fluent          |        🔴 Non-Fluent        |
| :--------------------------: | :-------------------------: |
| Smooth and consistent speech | Less fluent speech patterns |

</div>

The system processes speech recordings, extracts meaningful acoustic characteristics, selects the most informative features, and uses machine learning models to predict speech fluency.

To make the predictions understandable, the project also integrates **Explainable AI (XAI)** techniques using **SHAP** and **LIME**.

### 🎯 Main Goal

> **Convert a student's viva voice recording into an automated, interpretable fluency prediction using acoustic signal processing and machine learning.**

---

## ✨ Key Features

<div align="center">

<table>
<tr>
<td align="center">🎙️<br><b>Audio Processing</b><br><sub>Preprocess speech recordings</sub></td>
<td align="center">🎵<br><b>MFCC</b><br><sub>Extract spectral features</sub></td>
<td align="center">📈<br><b>LPC</b><br><sub>Model speech characteristics</sub></td>
</tr>

<tr>
<td align="center">📊<br><b>DCT</b><br><sub>Capture frequency information</sub></td>
<td align="center">🧩<br><b>Feature Fusion</b><br><sub>Combine acoustic features</sub></td>
<td align="center">🔎<br><b>Feature Selection</b><br><sub>Remove redundant features</sub></td>
</tr>

<tr>
<td align="center">🤖<br><b>ML Classification</b><br><sub>Compare multiple models</sub></td>
<td align="center">🔬<br><b>SHAP</b><br><sub>Global feature importance</sub></td>
<td align="center">💡<br><b>LIME</b><br><sub>Individual predictions</sub></td>
</tr>
</table>

</div>

---

## 🎬 Project Demo

<div align="center">

<img src="assets/demo.gif" alt="Project Demo" width="850"/>

<br>

<i>End-to-end speech fluency analysis pipeline</i>

</div>

> 💡 **Tip:** Add a short GIF showing audio input → preprocessing → feature extraction → prediction → explanation. This will make the repository immediately understandable to visitors.

---

# 🧠 How It Works

The project follows a complete machine learning pipeline:

<div align="center">

```text
🎙️ Student Viva Recording
             │
             ▼
┌────────────────────────────┐
│     Audio Preprocessing    │
│  Noise Reduction           │
│  Silence Removal           │
│  Normalization             │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│    Acoustic Feature        │
│       Extraction           │
│                            │
│   🎵 MFCC                  │
│   📈 LPC                   │
│   📊 DCT                   │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│      Feature Fusion        │
│     38-Dimensional         │
│      Representation        │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│    Feature Optimization    │
│                            │
│  Correlation Filtering     │
│  SFS                       │
│  PCA                       │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│     ML Classification      │
│                            │
│ Random Forest              │
│ Decision Tree              │
│ SVM                        │
│ kNN                        │
│ XGBoost                    │
│ Naive Bayes                │
│ Logistic Regression        │
│ MLP                        │
│ Stacking                   │
└─────────────┬──────────────┘
              │
              ▼
       🎯 Fluency Prediction
              │
        ┌─────┴─────┐
        ▼           ▼
   🟢 FLUENT    🔴 NON-FLUENT
              │
              ▼
       🔍 SHAP + LIME
        Explain Prediction
```

</div>

---

# 🎧 Audio Feature Extraction

The project combines three different acoustic feature representations.

<div align="center">

### 🎵 MFCC

**Mel-Frequency Cepstral Coefficients**

Captures perceptually important spectral characteristics of speech.

<br>

### 📈 LPC

**Linear Predictive Coding**

Models vocal-tract characteristics and contributes information related to speech smoothness.

<br>

### 📊 DCT

**Discrete Cosine Transform**

Provides compact frequency-domain information related to rhythm and energy distribution.

</div>

These feature groups are combined through **feature-level fusion**, resulting in a **38-dimensional acoustic representation**.

---

# 🔬 Feature Engineering

After feature extraction, the system performs several optimization steps.

<div align="center">

```text
38 Acoustic Features
        │
        ▼
Correlation Analysis
        │
        ▼
Remove Highly Correlated Features
        │
        ▼
Sequential Feature Selection
        │
        ▼
Principal Component Analysis
        │
        ▼
Retain 95% Variance
```

</div>

This helps reduce redundancy and improves the efficiency of the machine learning pipeline.

---

# 🤖 Machine Learning Models

The project compares several machine learning algorithms:

<div align="center">

| Model                  | Purpose                             |
| :--------------------- | :---------------------------------- |
| 🌲 Random Forest       | Ensemble tree-based classification  |
| 🌳 Decision Tree       | Interpretable classification        |
| 📍 kNN                 | Distance-based classification       |
| 📐 SVM                 | Margin-based classification         |
| ⚡ XGBoost              | Gradient boosting                   |
| 🧮 Naïve Bayes         | Probabilistic classification        |
| 📊 Logistic Regression | Linear classification               |
| 🧠 MLP                 | Neural-network based classification |
| 🔗 Stacking            | Ensemble of multiple classifiers    |

</div>

---

# 🏆 Results

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

<div align="center">

## 🥇 Best Model

### Random Forest

<table>
<tr>
<td align="center">
<h2>94.2%</h2>
<b>Accuracy</b>
</td>

<td align="center">
<h2>94.2%</h2>
<b>Precision</b>
</td>

<td align="center">
<h2>94.2%</h2>
<b>Recall</b>
</td>

<td align="center">
<h2>0.942</h2>
<b>F1-Score</b>
</td>
</tr>
</table>

</div>

### Model Comparison

| Model                | Test Accuracy | Test F1-Score |
| :------------------- | ------------: | ------------: |
| 🥇 **Random Forest** |     **94.2%** |     **0.942** |
| Naïve Bayes          |         92.7% |         0.927 |
| Decision Tree        |         91.1% |         0.911 |
| Stacking Ensemble    |         85.3% |         0.860 |
| XGBoost              |         82.8% |         0.828 |
| SVM                  |         74.7% |         0.810 |
| MLP                  |         72.1% |         0.799 |
| kNN                  |         77.2% |         0.772 |
| Logistic Regression  |         62.2% |         0.683 |

---

# 📊 Model Performance

<div align="center">

<img src="assets/model_comparison.png" width="850" alt="Model Comparison"/>

<br><br>

<img src="assets/confusion_matrix.png" width="650" alt="Random Forest Confusion Matrix"/>

</div>

### Random Forest Confusion Matrix

The Random Forest model correctly classified:

* **96 Non-Fluent** samples
* **143 Fluent** samples

with only:

* **4 Non-Fluent** samples misclassified
* **16 Fluent** samples misclassified

---

# 🔍 Explainable AI

A major component of this project is **interpretability**.

Instead of simply returning:

> `Prediction: Fluent`

the system can also provide information about **why the model made that prediction**.

---

## 🔬 SHAP

**SHAP — SHapley Additive exPlanations**

is used to understand the global importance of acoustic features.

<div align="center">

<img src="assets/shap_summary.png" width="800" alt="SHAP Feature Importance"/>

<br>

<i>SHAP-based feature importance for the Random Forest model</i>

</div>

The analysis highlighted LPC and MFCC features as important contributors to fluency classification.

---

## 💡 LIME

**LIME — Local Interpretable Model-Agnostic Explanations**

is used to explain individual predictions.

<div align="center">

<img src="assets/lime_explanation.png" width="800" alt="LIME Explanation"/>

<br>

<i>LIME-based explanation of an individual fluency prediction</i>

</div>

This makes the system more transparent and easier to interpret than a purely black-box prediction system.

---

# 🎞️ Visualization Gallery

<div align="center">

<table>
<tr>
<td align="center">
<img src="assets/waveform.gif" width="400"/><br>
<b>🎙️ Speech Waveform</b>
</td>

<td align="center">
<img src="assets/mfcc.gif" width="400"/><br>
<b>🎵 MFCC Visualization</b>
</td>
</tr>

<tr>
<td align="center">
<img src="assets/model_comparison.gif" width="400"/><br>
<b>📊 Model Comparison</b>
</td>

<td align="center">
<img src="assets/shap_summary.gif" width="400"/><br>
<b>🔍 SHAP Explanation</b>
</td>
</tr>
</table>

</div>

> **Note:** Replace these filenames with the actual GIF/image files you upload to your `assets/` folder.

---



# 🗂️ Dataset

The project uses student viva speech recordings labelled into two classes:

<div align="center">

### 🟢 Fluent

Smooth and relatively consistent speech.

### 🔴 Non-Fluent

Speech exhibiting lower fluency characteristics.

</div>

### Dataset Information

| Property                 | Details             |
| :----------------------- | :------------------ |
| 🎙️ Number of recordings | 1300                |
| ⏱️ Recording duration    | 10–40 seconds       |
| 🎚️ Sampling rate        | 16 kHz              |
| 📁 Format                | WAV                 |
| 🏷️ Classes              | Fluent / Non-Fluent |
| ⚖️ Class balancing       | SMOTE               |

> ⚠️ **Privacy:** Raw student recordings are not included in the public repository. The repository contains the processed feature dataset where applicable.

---

# 🛠️ Tech Stack

<div align="center">

<table>
<tr>
<td align="center"><b>🐍 Python</b><br>Core Development</td>
<td align="center"><b>🎧 Librosa</b><br>Audio Processing</td>
<td align="center"><b>🔢 NumPy</b><br>Numerical Computing</td>
</tr>

<tr>
<td align="center"><b>🐼 Pandas</b><br>Data Processing</td>
<td align="center"><b>🤖 Scikit-Learn</b><br>Machine Learning</td>
<td align="center"><b>⚡ XGBoost</b><br>Boosting</td>
</tr>

<tr>
<td align="center"><b>🔬 SHAP</b><br>Explainability</td>
<td align="center"><b>💡 LIME</b><br>Interpretability</td>
<td align="center"><b>📓 Jupyter</b><br>Experimentation</td>
</tr>
</table>

</div>

---

# 💡 What Makes This Project Different?

<div align="center">

<table>
<tr>
<td align="center">
🎧<br>
<b>Audio-Based</b><br>
<sub>Works directly with speech signals</sub>
</td>

<td align="center">
🧩<br>
<b>Feature Fusion</b><br>
<sub>MFCC + LPC + DCT</sub>
</td>

<td align="center">
🤖<br>
<b>ML-Based</b><br>
<sub>Multiple classifiers evaluated</sub>
</td>

<td align="center">
🔍<br>
<b>Explainable</b><br>
<sub>SHAP + LIME</sub>
</td>
</tr>
</table>

</div>

The project doesn't stop at classification. It combines **prediction with explanation**, allowing users to understand which acoustic characteristics influence the model's decision.

---

# 🔮 Future Improvements

* 🎤 Real-time microphone-based fluency detection
* 🌐 Web-based fluency evaluation application
* 📱 Mobile application integration
* 🧠 Transformer-based speech representations
* 🌍 Multilingual and accent-aware evaluation
* 📈 Student fluency progress tracking
* 📊 Automated feedback generation
* 🎯 Continuous fluency scoring instead of binary classification

---

# ⭐ Support

If you found this project useful or interesting:

<div align="center">

### ⭐ Star this repository

### 🍴 Fork the project

### 🐛 Report issues

### 💡 Suggest improvements

</div>

---

# 📜 Publication

<div align="center">

## 📄 Published IEEE Conference Paper

</div>

This project is based on the following published IEEE conference paper:

> **U. K. Kushal, K. M. Sai, R. Gnaneswar and P. B. Pati, "An Explainable Framework for Automated Speech Fluency Evaluation using Spectral Features with Statistical Classifiers," 2026 7th International Conference on Computational Intelligence and Networks (CINE), Bhubaneswar, India, 2026, pp. 1-6, doi: 10.1109/CINE68769.2026.11502875.**

<div align="center">

[![IEEE Paper](https://img.shields.io/badge/IEEE-Paper-00629B?style=for-the-badge\&logo=ieee)](https://doi.org/10.1109/CINE68769.2026.11502875)

</div>

---

<div align="center">

### 🎙️ Speech → Features → Machine Learning → Explainable Prediction

**Built with Python • Machine Learning • Audio Processing • Explainable AI**

<br>

⭐ **Fluency Analysis of Students Viva from Voice Signals**

</div>
