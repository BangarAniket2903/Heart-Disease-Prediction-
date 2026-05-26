import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("heart_disease_rf_pipeline.pkl")

# -----------------------------
# Header
# -----------------------------
st.title("❤️ Heart Disease Prediction System")

st.markdown("""
This application predicts the likelihood of **Heart Disease**
using a Machine Learning model trained on clinical patient data.

Fill in the patient information below and click **Predict Risk**.
""")

st.divider()

# -----------------------------
# Input Section
# -----------------------------
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=40
    )

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=0,
        max_value=700,
        value=200
    )

with col2:
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        ["N", "Y"]
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Risk", use_container_width=True):

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease Detected")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.metric(
        label="Heart Disease Probability",
        value=f"{probability:.2%}"
    )

    with st.expander("View Entered Patient Data"):
        st.dataframe(input_data, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Machine Learning Model: Tuned Random Forest Classifier | "
    "Built using Scikit-learn and Streamlit"
)