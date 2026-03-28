import streamlit as st
import numpy as np
import pickle
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Header ────────────────────────────────────────────────────────────────────
st.title("❤️ Heart Disease Prediction")
st.markdown(
    "Enter the patient's clinical details below to predict the likelihood of heart disease. "
    "This model uses a **Gaussian Naïve Bayes** classifier trained on the Heart Disease dataset."
)

if not model_loaded:
    st.error(
        "⚠️ `model.pkl` not found. Please place the trained model file in the same "
        "directory as `app.py` and restart the app."
    )
    st.stop()

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)

    sex = st.selectbox("Sex", options=["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0

    chest_pain_map = {
        "Typical Angina (ATA)": 0,
        "Atypical Angina (NAP)": 1,
        "Non-Anginal Pain (TA)": 2,
        "Asymptomatic (ASY)": 3,
    }
    chest_pain = st.selectbox("Chest Pain Type", options=list(chest_pain_map.keys()))
    chest_pain_val = chest_pain_map[chest_pain]

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120, step=1
    )

    cholesterol = st.number_input(
        "Serum Cholesterol (mg/dl)", min_value=0, max_value=700, value=200, step=1
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl", options=["No (0)", "Yes (1)"]
    )
    fasting_bs_val = 1 if fasting_bs.startswith("Yes") else 0

with col2:
    ecg_map = {
        "Normal": 1,
        "ST-T Wave Abnormality (ST)": 2,
        "Left Ventricular Hypertrophy (LVH)": 0,
    }
    resting_ecg = st.selectbox("Resting ECG Result", options=list(ecg_map.keys()))
    resting_ecg_val = ecg_map[resting_ecg]

    max_hr = st.number_input(
        "Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150, step=1
    )

    exercise_angina = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
    exercise_angina_val = 1 if exercise_angina == "Yes" else 0

    oldpeak = st.number_input(
        "Oldpeak (ST depression, numeric)", min_value=-10.0, max_value=10.0,
        value=1.0, step=0.1, format="%.1f"
    )

    slope_map = {"Up": 2, "Flat": 1, "Down": 0}
    st_slope = st.selectbox("ST Slope", options=list(slope_map.keys()))
    st_slope_val = slope_map[st_slope]

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    # The notebook applied LabelEncoder to every feature, which effectively
    # maps each unique value to an integer rank — for the raw numeric inputs
    # the encoded value equals the rank of the original value in sorted order.
    # For a real deployment re-fit the encoders and save them alongside the
    # model; here we pass raw numerics, which matches the single-sample
    # prediction cell in the notebook (cell 31).
    features = np.array([[
        age,
        sex_val,
        chest_pain_val,
        resting_bp,
        cholesterol,
        fasting_bs_val,
        resting_ecg_val,
        max_hr,
        exercise_angina_val,
        oldpeak,
        st_slope_val,
    ]])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    if prediction == 1:
        st.error(
            "### ⚠️ High Risk — Heart Disease Detected\n"
            "The model predicts the patient **has heart disease**. "
            "Please consult a cardiologist immediately."
        )
    else:
        st.success(
            "### ✅ Low Risk — No Heart Disease Detected\n"
            "The model predicts the patient is **healthy**. "
            "Continue regular health check-ups."
        )

    # Confidence bar
    confidence = max(proba) * 100
    st.metric("Model Confidence", f"{confidence:.1f}%")
    st.progress(int(confidence))

    st.caption(
        "⚕️ *This tool is for informational purposes only and does not replace "
        "professional medical advice.*"
    )

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Model:** Gaussian Naïve Bayes  
        **Dataset:** Heart Disease (UCI / Kaggle)  
        **Features used:** Age, Sex, Chest Pain Type, Resting BP,  
        Cholesterol, Fasting BS, Resting ECG, Max HR,  
        Exercise Angina, Oldpeak, ST Slope  
        """
    )
    st.divider()
    st.markdown("**How to run:**")
    st.code("streamlit run app.py", language="bash")
    st.markdown("Ensure `model.pkl` is in the same directory as `app.py`.")
