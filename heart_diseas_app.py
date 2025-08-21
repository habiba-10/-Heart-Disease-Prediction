import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction (Cleveland Dataset)",
    layout="centered"
)

st.title("Heart Disease Prediction System")
st.write(
    "This application predicts the likelihood of heart disease using a model trained on the UCI Cleveland dataset. "
    "Provide patient attributes below and click Predict."
)

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "heart_disease_model.pkl"
ENCODER_PATH = "encoder.pkl"
SELECTED_FEATURES_PATH = "selected_features.pkl"

# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    selected_features = joblib.load(SELECTED_FEATURES_PATH)
    return model, encoder, selected_features

model, encoder, selected_features = load_artifacts()

cat_cols = list(encoder.feature_names_in_)  # categorical columns
base_num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
eng_cols = ["chol_age_ratio", "thalach_oldpeak_ratio"]

# ----------------------------
# Mapping for user-friendly labels
# ----------------------------
category_mappings = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Pain",
        3: "Asymptomatic"
    },
    "fbs": {0: "Fasting Blood Sugar < 120 mg/dl", 1: "Fasting Blood Sugar ≥ 120 mg/dl"},
    "restecg": {
        0: "Resting ECG: Normal",
        1: "Resting ECG: ST-T Abnormality",
        2: "Resting ECG: Left Ventricular Hypertrophy"
    },
    "exang": {0: "Exercise Induced Angina: No", 1: "Exercise Induced Angina: Yes"},
    "slope": {
        0: "Slope of Peak Exercise ST Segment: Upsloping",
        1: "Slope of Peak Exercise ST Segment: Flat",
        2: "Slope of Peak Exercise ST Segment: Downsloping"
    },
    "ca": {
        0: "Number of Major Vessels Colored by Fluoroscopy: 0",
        1: "Number of Major Vessels Colored by Fluoroscopy: 1",
        2: "Number of Major Vessels Colored by Fluoroscopy: 2",
        3: "Number of Major Vessels Colored by Fluoroscopy: 3",
        4: "Number of Major Vessels Colored by Fluoroscopy: 4"
    },
    "thal": {
        3: "Thalassemia: Normal",
        6: "Thalassemia: Fixed Defect",
        7: "Thalassemia: Reversible Defect"
    }
}

# ----------------------------
# Helpers
# ----------------------------
def make_features_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    for col in base_num_cols + cat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["chol_age_ratio"] = df["chol"] / df["age"]
    df["thalach_oldpeak_ratio"] = df["thalach"] / (df["oldpeak"] + 1e-5)

    enc_arr = encoder.transform(df[cat_cols])
    enc_cols = encoder.get_feature_names_out(cat_cols)
    enc_df = pd.DataFrame(enc_arr, columns=enc_cols, index=df.index)

    all_features = pd.concat([df[base_num_cols + eng_cols], enc_df], axis=1)
    final_X = all_features.reindex(columns=selected_features, fill_value=0)

    return final_X

def predict_single(input_record: dict):
    raw_df = pd.DataFrame([input_record])
    X = make_features_from_raw(raw_df)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    return pred, proba

# ----------------------------
# Input Form
# ----------------------------
st.header("Patient Information")

with st.form("patient_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=250, value=130)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=700, value=240)
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    st.markdown("---")

    cat_inputs = {}
    for name, mapping in category_mappings.items():
        options = list(mapping.values())
        default_index = 0
        selected_label = st.selectbox(name, options=options, index=default_index)
        inv_map = {v: k for k, v in mapping.items()}  # reverse map
        cat_inputs[name] = inv_map[selected_label]

    submitted = st.form_submit_button("Predict")

# ----------------------------
# Prediction Output
# ----------------------------
if submitted:
    input_record = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "oldpeak": oldpeak,
        **cat_inputs
    }

    pred, proba = predict_single(input_record)
    percentage = proba * 100

    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f"High Risk of Heart Disease\nProbability: {percentage:.1f}%")
    else:
        st.success(f"No Heart Disease Predicted\nProbability: {percentage:.1f}%")