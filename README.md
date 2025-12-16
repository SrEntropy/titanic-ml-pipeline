# Titanic ML Pipeline

Exploring exploratory data analysis (EDA), feature engineering, and model evaluation on the classic Titanic dataset.  
This project is designed as a hands‑on learning resource for understanding how preprocessing choices and algorithm selection impact predictive performance.

---

## Project Overview
The goal is to predict passenger survival on the Titanic using machine learning.  
This matters because the dataset is a benchmark problem for classification, widely used to practice end‑to‑end ML workflows: data cleaning, feature engineering, model training, evaluation, and interpretability.

---

## Dataset Description
- **Source**: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)  
- **Key Features**:
  - Passenger demographics (Age, Sex, SibSp, Parch)
  - Ticket and cabin information
  - Socio‑economic indicators (Fare, Class)
  - Survival outcome (target variable)

---

## Pipeline Summary
1. **EDA** → Inspect missing values, distributions, correlations  
2. **Feature Engineering** → Encode categorical variables, impute missing data, create derived features  
3. **Modeling** → Train multiple ML algorithms (Logistic Regression, Decision Trees, Random Forests, Gradient Boosting)  
4. **Evaluation** → Compare models using accuracy, precision, recall, F1 score, and ROC‑AUC  

---

## ⚙️ How to Run
### Setup
```bash
pip install -r requirements.txt

