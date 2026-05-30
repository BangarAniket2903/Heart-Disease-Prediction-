# ❤️ Heart Disease Prediction System

## 📌 Overview

An end-to-end Machine Learning project that predicts the likelihood of heart disease using patient clinical data. The project covers data preprocessing, exploratory data analysis (EDA), model development, hyperparameter tuning, evaluation, and deployment using Streamlit.

---

## 🎯 Objective

Build a reliable classification model to identify patients at risk of heart disease and provide real-time predictions through a web application.

---

## 🛠️ Tech Stack

**Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Streamlit**

### Machine Learning Models

* Logistic Regression
* Decision Tree
* Random Forest
* KNN
* SVM

---

## 📊 Dataset

* **918 patient records**
* 11 clinical features
* Target: **HeartDisease (0 = No Disease, 1 = Disease)**

Key features include:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Max Heart Rate
* Exercise Angina
* ST Slope

---

## 🔍 Project Workflow

1. Data Cleaning & Missing Value Treatment
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training & Comparison
5. Hyperparameter Tuning (GridSearchCV)
6. Cross Validation
7. Model Deployment using Streamlit

---

## 📈 Model Performance

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------- | ------------- |
| Accuracy  | 89.13%              | 89.13%        |
| Precision | 89.42%              | 87.27%        |
| Recall    | 91.18%              | 94.12%        |
| F1 Score  | 90.29%              | 90.57%        |
| ROC-AUC   | 93.44%              | 93.03%        |

### 🏆 Final Model

**Random Forest Classifier**

Reason:

* Highest Recall (94.12%)
* Highest F1 Score (90.57%)
* Better suitability for healthcare prediction tasks

---

## 🚀 Deployment

The final model was deployed using **Streamlit**, enabling users to enter patient information and receive real-time heart disease predictions.

Run locally:

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Aniket Bangar**

Aspiring Data Scientist | Machine Learning Enthusiast
