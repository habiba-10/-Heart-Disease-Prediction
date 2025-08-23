# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Heart Disease Risk (Clinical Demo)",
    page_icon="❤",
    layout="centered"
)

# =========================
# CSS Styling
# =========================
st.markdown(
    """
    <style>
    .main { 
        background-color: #f6fbff; /* very light blue background */
        color: #1b2b3a; 
        font-family: "Segoe UI", sans-serif;
    }
    h1, h2, h3 { 
        color: #0066cc; 
        font-weight: 600;
    }
    label { font-weight: 500; color: #004080; }

    /* Form card */
    .form-card {
        background: #ffffff;
        border: 1px solid #cfe7fa;
        padding: 18px;
        border-radius: 16px;
        margin-bottom: 18px;
        box-shadow: 0px 3px 6px rgba(0,0,0,0.05);
    }

    /* Button */
    div.stButton > button {
        background: linear-gradient(90deg, #0099ff, #33ccff);
        color: white;
        padding: 10px 20px;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        font-size: 16px;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #0080e6, #1ab2ff);
    }

    /* Risk boxes */
    .risk-low { 
        background: #e6f9ff; border: 1px solid #66c2ff; padding: 14px;
        border-radius: 12px; color: #00334d; font-weight: 600;
    }
    .risk-med { 
        background: #fff9e6; border: 1px solid #e6c200; padding: 14px;
        border-radius: 12px; color: #664d00; font-weight: 600;
    }
    .risk-high { 
        background: #ffe6e6; border: 1px solid #e67373; padding: 14px;
        border-radius: 12px; color: #660000; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Load model bundle + encoder
# =========================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    bundle = joblib.load("heart_disease_model_bundle.pkl")
    enc = joblib.load("encoder.pkl")
    return bundle, enc

bundle, encoder = load_artifacts()
final_model = bundle["model"]
best_threshold = bundle.get("threshold", 0.5)
hybrid_features = bundle["features"]

CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

def build_features(user):
    num_df = pd.DataFrame([{
        "age": user["age"],
        "trestbps": user["trestbps"],
        "chol": user["chol"],
        "thalach": user["thalach"],
        "oldpeak": user["oldpeak"],
    }])

    cat_df = pd.DataFrame([{
        "sex": user["sex"],
        "cp": float(user["cp"]),
        "fbs": user["fbs"],
        "restecg": float(user["restecg"]),
        "exang": user["exang"],
        "slope": float(user["slope"]),
        "ca": float(user["ca"]),
        "thal": float(user["thal"])
    }])

    encoded = encoder.transform(cat_df)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(CAT_COLS))

    df_model = pd.concat([num_df, encoded_df], axis=1)
    df_model["thalach_oldpeak_ratio"] = df_model["thalach"] / (df_model["oldpeak"] + 1e-5)

    for col in hybrid_features:
        if col not in df_model.columns:
            df_model[col] = 0.0

    X = df_model[hybrid_features].astype(float)
    return X

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.title("ℹ About the App")
    st.markdown(
        f"""
        This demo predicts *heart disease risk* using your trained ensemble model.  
        
        *Threshold*
        - Default decision threshold: *{best_threshold:.2f}*
        - You can adjust it below.
        """
    )
    user_threshold = st.slider(
        "Decision Threshold",
        min_value=0.10, max_value=0.90, value=float(best_threshold), step=0.01
    )

# =========================
# Main Title
# =========================
st.title("❤ Heart Disease Risk Prediction")
st.markdown("Enter patient details below:")

# =========================
# Form Layout (2 columns with cards)
# =========================
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        age = st.number_input("Age (years)", 20, 100, 45)
        sex = st.selectbox("Sex", [("Male",1.0),("Female",0.0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox("Chest Pain Type", [("Typical Angina",1.0),("Atypical Angina",2.0),("Non-anginal",3.0),("Asymptomatic",4.0)], format_func=lambda x: x[0])[1]
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        exang = st.selectbox("Exercise Induced Angina", [("No",0.0),("Yes",1.0)], format_func=lambda x: x[0])[1]
        slope = st.selectbox("Slope of ST Segment", [("Upsloping",1.0),("Flat",2.0),("Downsloping",3.0)], format_func=lambda x: x[0])[1]
        ca = st.selectbox("Major Vessels (0-3)", [0.0,1.0,2.0,3.0])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        thal = st.selectbox("Thalassemia", [("Normal",3.0),("Fixed Defect",6.0),("Reversible Defect",7.0)], format_func=lambda x: x[0])[1]
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [("No",0.0),("Yes",1.0)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox("Resting ECG", [("Normal",0.0),("ST-T Abnormality",1.0),("LV Hypertrophy",2.0)], format_func=lambda x: x[0])[1]
        st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("Predict Risk")

# =========================
# Prediction
# =========================
if submitted:
    user = {
        "age": float(age),
        "sex": float(sex),
        "cp": float(cp),
        "trestbps": float(trestbps),
        "chol": float(chol),
        "thalach": float(thalach),
        "oldpeak": float(oldpeak),
        "fbs": float(fbs),
        "restecg": float(restecg),
        "exang": float(exang),
        "slope": float(slope),
        "ca": float(ca),
        "thal": float(thal)
    }

    X = build_features(user)
    prob = float(final_model.predict_proba(X)[0, 1])
    pred = int(prob >= user_threshold)

    if prob < 0.33:
        risk_label = "Low Risk"
        risk_class = "risk-low"
    elif prob < 0.66:
        risk_label = "Medium Risk"
        risk_class = "risk-med"
    else:
        risk_label = "High Risk"
        risk_class = "risk-high"

    st.markdown("### Result")
    st.progress(prob)
    st.markdown(f"*Predicted Probability:* {prob:.2%}")
    st.markdown(f"<div class='{risk_class}'><b>Decision:</b> {'Positive (Disease)' if pred else 'Negative (No Disease)'} — threshold={user_threshold:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"*Risk Category:* {risk_label}")