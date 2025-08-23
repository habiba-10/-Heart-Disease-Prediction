## Heart Disease Risk Prediction - ML Pipeline & Streamlit App

📌 Project Overview

This project is a complete machine learning pipeline for predicting heart disease risk using the UCI Heart Disease Dataset (Cleveland).
It integrates data preprocessing, exploratory data analysis (EDA), feature engineering, feature selection, model training, hyperparameter tuning, ensemble learning, and deployment with Streamlit.

The app provides real-time risk predictions for patients based on clinical features.


---

🚀 Steps Implemented

1. Data Preprocessing

Handled missing values, duplicates, and outliers (clipping numeric columns).

Encoded categorical variables via One-Hot Encoding.

Scaled features using StandardScaler & MinMaxScaler.


2. Exploratory Data Analysis (EDA)

Correlation heatmaps, countplots, barplots, and histograms for key features.

Identified relationships between clinical variables and heart disease outcomes.


3. Feature Engineering

Added interaction features:

thalach_oldpeak_ratio = thalach / (oldpeak + 1e-5)

chol_age_ratio = chol / age


Combined feature selection methods:

Random Forest importance, Chi-Square, and RFE for robust hybrid selection.



4. Dimensionality Reduction

Applied PCA to assess variance and visualize features.

Retained minimal components while maintaining 95% explained variance.


5. Handling Class Imbalance

Used SMOTE for oversampling minority class in the training set.


6. Model Training & Hyperparameter Tuning

Trained models:

Logistic Regression

Decision Tree

Random Forest (tuned)

SVM

XGBoost (tuned)


Hyperparameters optimized using RandomizedSearchCV.

Evaluated with cross-validation accuracy, precision, recall, F1-score, and ROC-AUC.



7. Ensemble Learning

Combined XGBoost + Random Forest in a Voting Classifier (soft voting).

Optionally calibrated probabilities with CalibratedClassifierCV.


8. Threshold Optimization

Calculated optimal decision threshold using Youden’s J statistic for better sensitivity-specificity balance.


9. Deployment

Streamlit app with interactive forms for patient data.

Displays predicted probability, decision, and risk category (Low/Medium/High).





---

📊 Model Performance

✅ Final Voting Ensemble (XGBoost + RF)

Balanced Accuracy: 0.924

ROC AUC: 0.958

Optimal Threshold: 0.5


Classification Report @ Optimal Threshold:

Class	Precision	Recall	F1-score	Support

0 (No Disease)	1.00	0.85	0.92	33
1 (Disease)	0.85	1.00	0.92	28



---

🔍 Individual Model Cross-Validation Accuracy

Logistic Regression: 0.798 ± 0.049

Decision Tree: 0.718 ± 0.038

Random Forest: 0.782 ± 0.050

SVM: 0.630 ± 0.032

XGBoost: 0.771 ± 0.034



---

🧩 Final Selected Features

oldpeak

cp_2.0, cp_3.0, cp_4.0

exang_1.0

slope_2.0

ca_1.0, ca_2.0

thal_7.0

thalach

thalach_oldpeak_ratio

age



---

🛠 Technologies Used

Python Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost, joblib

Deployment: Streamlit

Modeling: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, Voting Ensemble



---

---

# How to Run Project

1. *استنساخ المشروع*
git clone https://github.com/habiba-10/-Heart-Disease-Prediction.git
cd -Heart-Disease-Prediction

2. *إنشاء وتفعيل البيئة الافتراضية*
python -m venv venv
# لتفعيل البيئة على ويندوز
venv\Scripts\activate
# لتفعيل البيئة على ماك/لينكس
source venv/bin/activate

3. *تثبيت المكتبات المطلوبة*
pip install -r requirements.txt

4. *تشغيل التطبيق محليًا باستخدام Streamlit*
streamlit run heart_disease_app.py

5. *الوصول للتطبيق مباشرة عبر الإنترنت*
افتح هذا اللينك في المتصفح:
https://heart-health-predict.streamlit.app/

6. *استخدام التطبيق*
- املي بيانات المريض في الفورم
- اضغطي على *Predict* عشان يظهرلك احتمال إصابة المريض بأمراض القلب مع النسبة المئوية

