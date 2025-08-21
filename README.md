# 🫀 Heart Disease Prediction - Machine Learning Pipeline

## 📌 Project Overview
This project aims to build a *comprehensive machine learning pipeline* for predicting the likelihood of heart disease using the *UCI Heart Disease Dataset*.  
The pipeline covers *data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and model evaluation, followed by deployment using **Streamlit*.

---

## 🚀 Steps Implemented
1. *Data Preprocessing*
   - Handling missing values, duplicates, and outliers.
   - Encoding categorical variables (One-Hot Encoding).
   - Feature scaling using StandardScaler & MinMaxScaler.

2. *Exploratory Data Analysis (EDA)*
   - Visualizations with matplotlib & seaborn (correlation heatmaps, distributions, boxplots).
   - Outlier detection and insights on key features.

3. *Feature Engineering*
   - Created interaction features such as thalach_oldpeak_ratio.
   - Selected top contributing features for the final model.

4. *Model Training & Evaluation*
   - Trained multiple models: 
     - Logistic Regression  
     - Decision Tree  
     - Random Forest  
     - Support Vector Machine (SVM)  
     - XGBoost  
   - Evaluated using *Cross-Validation Accuracy*, Precision, Recall, F1-Score, ROC-AUC.

5. *Ensemble Learning*
   - Final model is a *Voting Classifier* combining *Random Forest* and *XGBoost* for better performance.

6. *Deployment*
   - Built a *Streamlit web app* for real-time heart disease prediction.
   - Input form for patient medical details with instant prediction output.

---

## 📊 Model Performance

### ✅ Final Ensemble Model (Voting XGBoost + RF)
- *Accuracy:* 0.84  
- *Balanced Accuracy:* 0.832  
- *ROC AUC:* 0.944  

*Classification Report:*
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Disease) | 0.83 | 0.88 | 0.85 | 33 |
| 1 (Disease)    | 0.85 | 0.79 | 0.81 | 28 |

*Macro Avg:* Precision = 0.84 | Recall = 0.83 | F1-score = 0.83  
*Weighted Avg:* Precision = 0.84 | Recall = 0.84 | F1-score = 0.84  

---

### 🔍 Individual Model Cross-Validation Accuracy
- Logistic Regression: *0.817 ± 0.031*  
- Decision Tree: *0.748 ± 0.043*  
- Random Forest: *0.821 ± 0.052*  
- SVM: *0.630 ± 0.032*  
- XGBoost: *0.798 ± 0.070*  

---

## 🧩 Final Selected Features
The following features were selected for the deployed Streamlit app:
- oldpeak  
- cp_2.0, cp_3.0, cp_4.0  
- exang_1.0  
- slope_2.0  
- ca_1.0, ca_2.0  
- thal_7.0  
- thalach_oldpeak_ratio  

---

## 🛠 Technologies Used
- *Python Libraries:* pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, joblib  
- *Deployment:* streamlit  
- *Modeling:* Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, Ensemble Learning  

---




