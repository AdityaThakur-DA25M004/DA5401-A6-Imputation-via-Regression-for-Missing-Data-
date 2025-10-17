# 🎓 Assignment 6 – *Imputation via Regression*

#### Name: Aditya Thakur &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Roll No: DA25M004

> **Course:** DA5401 – Data Analysis and Machine Learning  
> **Objective:** To explore and compare multiple imputation strategies — Median, Linear Regression, and Decision Tree Regression — and evaluate their impact on model performance using Logistic Regression.  


---

## 🧾 Introduction

This notebook focuses on analyzing the effectiveness of different **data imputation strategies** for handling missing values in a credit default dataset.

Two main **approaches** were implemented:

1. **Approach 1 — Using `df_original`**  
   - Started from the clean dataset (no missing values).  
   - Artificially introduced missing values **only in one column (`AGE`)**.  
   - Created two imputed datasets:
     - `df_B₁` → Linear Regression Imputation  
     - `df_C₁` → Decision Tree Regression Imputation  

2. **Approach 2 — Using `df_missing`**  
   - Introduced **Missing At Random (MAR)** values (5–10%) in multiple numeric columns such as `AGE`, `BILL_AMT1`, and `PAY_AMT1`.  
   - Created three imputed datasets:
     - `df_A` → Median Imputation  
     - `df_B` → Linear Regression Imputation (AGE column)  
     - `df_C` → Decision Tree Regression Imputation (AGE column)  
     - `df_D` → Listwise Deletion (rows with missing values removed)

Since both approaches yielded nearly identical model performance, the **further analysis** is based on **Approach 2 (`df_missing`)**.

---

## 🧩 Datasets Created

| Dataset | Source | Imputation Technique | Description |
|----------|---------|----------------------|--------------|
| **A** | `df_missing` | Median Imputation | Filled all numeric missing values with median. |
| **B** | `df_missing` | Linear Regression | Predicted missing `AGE` using Linear Regression. |
| **C** | `df_missing` | Decision Tree Regression | Predicted missing `AGE` using Decision Tree Regressor. |
| **D** | `df_missing` | Listwise Deletion | Dropped rows containing any missing values. |
| **B₁** | `df_original` | Linear Regression | Introduced NaNs in `AGE` only, imputed with Linear Regression. |
| **C₁** | `df_original` | Decision Tree Regression | Introduced NaNs in `AGE` only, imputed with Decision Tree Regressor. |

---

## ⚙️ Step-by-Step Workflow

### **Step 1 — Data Loading and Exploration**
- Loaded the **credit default dataset** containing demographic and financial features.
- Checked for duplicates, data types, and basic statistics.
- Identified numeric and categorical columns.

### **Step 2 — Introducing Missingness**
- Simulated **MAR (Missing At Random)** missing values (5–10%) in 2–3 numeric columns (`AGE`, `BILL_AMT1`, `PAY_AMT1`).
- Visualized the missingness pattern using a **heatmap**.
- Stored the modified data as `df_missing`.

### **Step 3 — Creating Imputed Datasets**
#### 🟩 Dataset A — Median Imputation
- Filled missing values using the **median** of each numeric column.
- Explained why median is preferred over mean (robust to skewness and outliers).

#### 🟦 Dataset B / B₁ — Linear Regression Imputation
- Used predictors (excluding `ID`, `AGE`, and target).
- Split into `df_known` (with AGE) and `df_missing_age` (without AGE).
- Trained a **Linear Regression model** on `df_known` and predicted missing `AGE` values.
- Applied **StandardScaler** before training to ensure model stability.

#### 🟧 Dataset C / C₁ — Decision Tree Regression Imputation
- Used the same predictors and missing indices as in B/B₁.
- Trained a **Decision Tree Regressor (max_depth=8)** to capture non-linear relations.
- Predicted missing `AGE` and replaced in dataset.

#### 🟥 Dataset D — Listwise Deletion
- Removed all rows containing NaN values from `df_missing`.

---

## 🧮 Unified Processing Pipeline

A **modular pipeline** was created for all datasets (A, B, C, D):

### 🔹 Preprocessing
- `preprocess_datasets()` → fills missing values for A, B, C; creates D via listwise deletion.

### 🔹 Data Splitting
- `split_data()` → performs 80–20 stratified split into train and test sets.

### 🔹 Feature Scaling
- `standardize_data()` → applies `StandardScaler` for normalization.

### 🔹 Dimensionality Reduction
- `apply_pca()` → retains 98% variance using Principal Component Analysis (PCA).

### 🔹 Model Training & Evaluation
- `train_and_evaluate()` → trains `LogisticRegression` (class_weight='balanced') for each dataset.  
- Outputs full **classification report**.

### 🔹 Metrics & Curves
- `extract_classwise_metrics()` → computes Precision, Recall, F1 per class.  
- `compute_curve_data()` → generates Precision–Recall and ROC curves with AUC.

---

## 📊 Results Summary

| Dataset | ROC-AUC | PR-AUC | Observation |
|----------|----------|--------|--------------|
| **A (Median)** | 0.708 | 0.491 | Baseline; stable performance |
| **B (Linear)** | 0.707 | 0.491 | Comparable to median |
| **C (Decision Tree)** | 0.707 | 0.491 | Slightly smoother imputation |
| **D (Listwise Deletion)** | 0.709 | 0.490 | Slight drop due to data loss |

**Observation:**  
Results across A, B, and C are nearly identical, showing that imputing a single variable (AGE) has minimal effect on model performance dominated by other strong predictors (credit limits, payment history, bill amounts).

---
## 📈 Visualizations
1. **Missing Value Heatmap** — visual confirmation of MAR missingness.  
2. **Histograms** — AGE distribution before and after imputation (Linear & Tree).  
3. **Bar Charts** — Class-wise Precision, Recall, and F1-score comparisons.  
4. **Precision–Recall and ROC Curves** — Comparative model performance across all datasets.

---

## 🧠 Key Insights
- **Logistic Regression** performance remains stable across all imputations due to strong predictive influence from multiple correlated features such as **payment history, bill amounts, and credit limits**.  
- **Listwise Deletion (D)** performs worse due to **loss of data**, reinforcing that dropping missing rows is not an effective strategy.  
- **Decision Tree Imputation (C / C₁)** better captures **non-linear patterns** between predictors and the target feature (`AGE`), making it conceptually preferable for this dataset.  
- Although results across all models appear similar, Decision Tree Imputation aligns more naturally with datasets containing **discrete thresholds** and **heterogeneous relationships**, like financial data.

---

## 🧩 Final Conclusion

The comparative analysis shows that **all imputation strategies (Median, Linear, and Decision Tree)** lead to very similar classification outcomes when evaluated with **Logistic Regression**.  
This occurs because the model relies heavily on strong predictor variables such as repayment history and billing amounts, reducing the sensitivity to imputed variations in a single feature (`AGE`).  

However, from a **conceptual and data-driven perspective**:

- **Listwise Deletion** is least recommended, as it reduces dataset size and diversity.  
- **Median Imputation** offers simplicity and stability for moderately skewed features.  
- **Linear Regression Imputation** performs effectively when the relationship between features is linear.  
- **Decision Tree Imputation** provides a flexible, non-linear alternative that performs well without strict statistical assumptions.  

Thus, for practical applications where relationships among variables are complex and potentially non-linear — as is typical in credit risk datasets — **Decision Tree Imputation** is a robust and preferred choice.  

Overall, since **Approach 1** (original dataset) and **Approach 2** (MAR dataset) yield almost identical performance, the remainder of the study and model evaluations are conducted using **Approach 2** based on `df_missing`.

---
