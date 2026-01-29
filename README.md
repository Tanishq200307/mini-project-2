# Loan Approval Prediction – EDA & Modeling

## 📌 Project Overview
This project analyzes a loan application dataset to understand the factors influencing **loan approval decisions** and to build predictive models for loan approval.  

The work is divided into:
1. **Exploratory Data Analysis (EDA)** – understanding data structure, distributions, missing values, and class imbalance  
2. **Data Preprocessing** – handling missing values, scaling, and encoding  
3. **Modeling & Evaluation** – comparing multiple classifiers with imbalance-handling techniques  
4. **Conclusion & Best Model Selection**

---

## 📂 Project Structure

```text
mini-project-II/
├── data/
│   └── train_u6lujuX_CVtuZ9i.csv        # Dataset used for analysis
│
├── notebooks/
│   ├── 01_exploration.ipynb             # Data exploration and visualization
│   └── 02_modeling.ipynb                # Model building and evaluation
│
├── requirements.txt                     # Project dependencies
├── .gitignore                           # Git ignore rules
└── README.md                            # Project documentation

```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 📊 Dataset Description
- **Dataset Size:** 614 records, 13 features  
- **Target Variable:** `Loan_Status`
  - `Y` → Loan Approved  
  - `N` → Loan Rejected  

### Key Features
- Demographics: `Gender`, `Married`, `Dependents`, `Education`
- Employment: `Self_Employed`
- Financial: `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`
- Loan Details: `Loan_Amount_Term`
- Credit: `Credit_History`
- Location: `Property_Area`

---

## 🔍 Exploratory Data Analysis (EDA)

### 1️⃣ Data Types & Missing Values
- Dataset contains **numerical and categorical features**
- Missing values were found in:
  - `Credit_History` (~8.1%)
  - `Self_Employed` (~5.2%)
  - `LoanAmount` (~3.6%)
  - `Dependents`, `Loan_Amount_Term`, `Gender`, `Married`

➡️ **Conclusion:** Missing values must be handled carefully to avoid bias.

---

### 2️⃣ Target Variable Distribution
- **Approved (Y): ~68.7%**
- **Rejected (N): ~31.3%**

➡️ **Insight:** The dataset is **moderately imbalanced**, which justifies:
- Using `class_weight="balanced"`
- Applying **SMOTE** for oversampling

---

### 3️⃣ Key Observations from EDA
- Loan approvals are **strongly correlated with Credit History**
- Income and loan amounts show **right-skewed distributions**
- Categorical variables (Education, Property Area, Employment) provide meaningful separation
- Class imbalance could negatively affect recall for rejected loans if not addressed

---

## 🛠️ Data Preprocessing
- **Numerical Features**
  - Median imputation
  - Standard scaling
- **Categorical Features**
  - Most-frequent imputation
  - One-hot encoding
- **Train/Test Split**
  - 80% training, 20% testing
  - Stratified by target variable

---

## 🤖 Models Trained
The following models were evaluated:

1. **Logistic Regression**
   - With `class_weight="balanced"`
2. **Random Forest**
   - With `class_weight="balanced"`
3. **Logistic Regression + SMOTE**
4. **Random Forest + SMOTE**
5. **Tuned Logistic Regression + SMOTE**
   - Hyperparameter tuning using **GridSearchCV**
   - Optimized for **F1-score**

---

## 📈 Model Evaluation Metrics
- Precision
- Recall
- F1-score
- ROC–AUC
- Confusion Matrix
- ROC Curves

---

## 🏆 Model Comparison Summary

The table below compares all trained models using **Precision, Recall, F1-score, and ROC-AUC**.  
Given the class imbalance in the dataset, **F1-score and ROC-AUC** were prioritized for model evaluation.

| Model | Precision | Recall | F1 | ROC-AUC |
|------|----------|--------|----|--------|
| **Logistic Regression + SMOTE** | **0.86** | **0.96** | **0.91** | 0.87 |
| Tuned Logistic Regression + SMOTE (GridSearchCV) | 0.84 | **0.99** | 0.91 | **0.87** |
| Random Forest + SMOTE | 0.84 | 0.98 | 0.90 | 0.80 |
| Random Forest (class_weight=balanced) | 0.84 | 0.98 | 0.90 | 0.79 |
| SVC (class_weight=balanced) | 0.84 | 0.95 | 0.90 | 0.83 |
| Logistic Regression (class_weight=balanced) | 0.86 | 0.93 | 0.89 | 0.86 |
| SVC + SMOTE | 0.84 | 0.95 | 0.89 | 0.85 |

---

### 🔍 Key Insights

- **Logistic Regression + SMOTE** achieved the **best overall F1-score (0.91)**, offering the strongest balance between precision and recall.
- **Hyperparameter tuning (GridSearchCV)** slightly improved recall (**0.99**) and delivered the **highest ROC-AUC (0.87)**.
- **Random Forest models** achieved very high recall but showed **lower ROC-AUC**, indicating weaker probability calibration.
- **Support Vector Classifier (SVC)** performed competitively, with **F1 ≈ 0.90**, but **SMOTE did not significantly improve SVC performance**.
- Overall, **SMOTE was most effective for linear models**, particularly Logistic Regression, in handling class imbalance.

---

### ✅ Final Model Selection

Based on the evaluation metrics, **Logistic Regression with SMOTE** was selected as the final model due to its:
- Highest F1-score
- Strong ROC-AUC
- Simplicity and interpretability
- Robust performance on imbalanced data


---

## 🔑 Key Conclusions
- **Credit history** is the most influential feature for loan approval
- Handling **class imbalance** significantly improves model performance
- **SMOTE + Logistic Regression** outperforms more complex models like Random Forest
- Simpler, well-regularized models generalize better for this dataset
- F1-score is a more appropriate metric than accuracy due to imbalance

---

## 🚀 Future Improvements
- Cost-sensitive threshold tuning for business-specific risk
- Feature importance analysis & SHAP explainability
- Deployment as a REST API or web application
- Model monitoring & retraining pipeline

---



## 👤 Authors
**Tanishq Rawat** 

**Aristide Kanamugire**

