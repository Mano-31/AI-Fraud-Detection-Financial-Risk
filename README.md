# 💳 AI-Powered Fraud Detection & Financial Risk Intelligence System

An end-to-end Machine Learning project that detects fraudulent credit card transactions using **XGBoost**, **SMOTE**, and **Feature Engineering**, with an interactive **Streamlit Dashboard** for fraud detection, risk analysis, and transaction monitoring.

---
## 🚀 Live Demo

Try the deployed AI Fraud Detection Dashboard:

👉 (https://ai-fraud-detection-financial-risk.streamlit.app)

## 📌 Project Overview

Financial fraud is a major challenge because fraudulent transactions are extremely rare compared to genuine transactions.

This project develops a machine learning-based fraud detection system that:

* Detects fraudulent credit card transactions
* Handles highly imbalanced data using SMOTE
* Performs feature engineering
* Compares multiple machine learning models
* Uses XGBoost for fraud prediction
* Performs hyperparameter tuning
* Evaluates models using Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix
* Builds an end-to-end ML pipeline
* Saves the trained ML pipeline using Joblib
* Provides an interactive Streamlit dashboard
* Supports single-transaction fraud prediction
* Classifies transactions into different risk levels

---

## 🎯 Project Objectives

* Build an automated credit card fraud detection system
* Handle severe class imbalance
* Improve fraud detection performance
* Reduce false-positive predictions
* Provide fraud probability
* Classify transactions based on risk level
* Build an interactive fraud monitoring dashboard
* Create a reusable machine learning pipeline
* Prepare the project for deployment

---

## 🧠 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
StandardScaler
   ↓
SMOTE
   ↓
Model Training
   ↓
Logistic Regression
   ↓
Random Forest
   ↓
XGBoost
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Final XGBoost Pipeline
   ↓
Model Saving
   ↓
Streamlit Dashboard
```

---

## 📊 Dataset

This project uses the **Credit Card Fraud Detection Dataset**.

The dataset contains anonymized credit card transactions.

| Property               | Details   |
| ---------------------- | --------- |
| Total Transactions     | 284,807   |
| Features               | 30        |
| Target Variable        | Class     |
| Genuine Transaction    | Class = 0 |
| Fraudulent Transaction | Class = 1 |

The dataset is highly imbalanced because fraudulent transactions represent a very small percentage of all transactions.

The original dataset is not included in this repository because of its large size.

### Dataset Source

[Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-learn

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Model Management

* Joblib

### Deployment

* Streamlit

---

## 🔧 Feature Engineering

Two additional features were created from the original transaction data.

### 1. Log Transformation of Amount

```python
data["Amount_log"] = np.log1p(data["Amount"])
```

This transformation reduces the effect of highly skewed transaction amounts.

### 2. Time Conversion

```python
data["Time_hours"] = data["Time"] / 3600
```

This converts transaction time from seconds into hours.

---

## ⚖️ Handling Class Imbalance

The dataset contains very few fraudulent transactions compared with genuine transactions.

To handle this severe class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)
```

SMOTE was applied only to the training data to avoid data leakage into the test set.

---

## 🤖 Machine Learning Models

Several classification algorithms were evaluated.

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Random Forest

Used to capture nonlinear relationships between transaction features.

### 3. XGBoost

Used as the main high-performance machine learning model.

### 4. Tuned XGBoost

XGBoost hyperparameters were optimized using **RandomizedSearchCV**.

The final model was selected mainly based on fraud detection performance and F1-score.

---

## 📈 Model Evaluation

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

For fraud detection, **Precision, Recall, F1-Score, and ROC-AUC** are especially important because the dataset is highly imbalanced.

---

## 🏆 Model Results

| Model               | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | --------: | -----: | -------: | ------: |
| Logistic Regression |      0.14 |   0.85 |     0.24 |  0.9600 |
| Random Forest       |      0.50 |   0.82 |     0.62 |  0.9695 |
| XGBoost             |      0.64 |   0.80 |     0.71 |  0.9737 |
| Tuned XGBoost       |  **0.88** |   0.80 | **0.84** |  0.9723 |

The **Tuned XGBoost** model achieved the best F1-score and precision among the evaluated models.

---

## 🔄 ML Pipeline

The final machine learning pipeline contains:

```text
StandardScaler
      ↓
SMOTE
      ↓
XGBoost
```

This ensures that preprocessing, class balancing, and prediction are handled consistently.

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

final_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.2,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    ))
])
```

---

## 💾 Model Saving

The trained machine learning pipeline is saved using Joblib.

```python
joblib.dump(
    final_pipeline,
    "models/fraud_detection_pipeline.pkl"
)
```

The saved model can later be loaded for prediction.

```python
model = joblib.load(
    "models/fraud_detection_pipeline.pkl"
)
```

---

## 📊 Streamlit Dashboard

The project includes an interactive Streamlit dashboard for fraud detection, risk analysis, and transaction monitoring.

### 🖥️ Dashboard Preview

![Fraud Detection Dashboard](https://raw.githubusercontent.com/Mano-31/AI-Fraud-Detection-Financial-Risk/main/images/dashboard.png)

### Dashboard Features

* 💳 Total transaction count
* 🟢 Genuine transaction count
* 🚨 Fraud transaction count
* 🔴 High-risk transaction count
* 📈 Genuine vs Fraud chart
* ⚠️ Risk distribution
* 📈 ROC-AUC curve
* 📊 Confusion Matrix
* 🔍 Prediction results
* 🚨 Fraud transaction table
* 🔴 High-risk transaction table
* ⬇️ Download prediction results
* 🔎 Single transaction prediction

---

## 🚦 Risk Classification

The system assigns risk levels based on fraud probability.

| Fraud Probability | Risk Level     |
| ----------------- | -------------- |
| < 50%             | 🟢 Low Risk    |
| 50% – 79.99%      | 🟠 Medium Risk |
| ≥ 80%             | 🔴 High Risk   |

---

## 🔎 Single Transaction Prediction

The Streamlit application allows users to manually enter:

* Transaction Time
* Transaction Amount
* V1–V28 features

The system returns:

* Prediction
* Fraud Probability
* Risk Level

### Example

```text
Prediction: Genuine
Fraud Probability: 2.15%
Risk Level: 🟢 Low Risk
```

---

## 📁 Project Structure

```text
Fraud_Detection_AI/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   └── fraud_detection_pipeline.pkl
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── images/
│   └── dashboard.png
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Mano-31/AI-Fraud-Detection-Financial-Risk.git
```

### 2. Navigate to the Project

```bash
cd AI-Fraud-Detection-Financial-Risk
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser:

```text
http://localhost:8501
```

---

## 📂 Dataset Setup

Download the **Credit Card Fraud Detection Dataset** from Kaggle.

Place the downloaded file:

```text
creditcard.csv
```

inside:

```text
data/
```

Expected path:

```text
data/creditcard.csv
```

The dataset is excluded from GitHub using `.gitignore`.

---

## 📦 Requirements

The project requires:

```text
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
joblib
streamlit
matplotlib
seaborn
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

* Real-time fraud detection
* Real-time transaction monitoring
* SHAP-based model explainability
* Database integration
* Email/SMS fraud alerts
* User authentication
* Cloud deployment
* Model monitoring
* Automated model retraining
* REST API-based prediction service

---

## ⚠️ Disclaimer

This project is developed for **educational and demonstration purposes**.

A production-level financial fraud detection system would require additional security controls, real-time monitoring, model validation, explainability, regulatory compliance, and extensive testing.

---

## 👨‍💻 Author

**Manogaran P**

B.Tech Information Technology

### Interests

* Data Science
* Machine Learning
* Data Analytics
* Artificial Intelligence

---

⭐ If you find this project useful, consider giving the repository a star!
