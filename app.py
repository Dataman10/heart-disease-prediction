import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
    <style>

    .main {
        background-color: #0E1117;
    }

    h1 {
        color: #FF4B4B;
        text-align: center;
        font-size: 50px;
    }

    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-size: 20px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #ff1e1e;
        color: white;
    }

    .css-1d391kg {
        padding-top: 2rem;
    }

    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL FILES ---------------- #

model = joblib.load('LogReg_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

# ---------------- TITLE ---------------- #

st.title("❤️ Heart Disease Prediction")

st.markdown("""
### Predict Heart Disease Risk Using Machine Learning

Provide the patient's medical information below.
""")

st.divider()

# ---------------- USER INPUTS ---------------- #

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)

    sex = st.selectbox(
        "Sex",
        ['M', 'F']
    )

    pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    RestingBP = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        90,
        200,
        120
    )

    Cholesterol = st.number_input(
        "Cholesterol (mg/DL)",
        100,
        600,
        200
    )

with col2:

    FastingBS = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )

    RestingECG = st.selectbox(
        "Resting ECG",
        ['Normal', 'ST', 'LVH']
    )

    MaxHR = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        100
    )

    ExerciseAngina = st.selectbox(
        "Exercise-Induced Angina",
        ['Y', 'N']
    )

    Oldpeak = st.slider(
        "Oldpeak ST Depression",
        0.0,
        6.0,
        1.0
    )

    ST_Slope = st.selectbox(
        "ST Slope",
        ['Up', 'Flat', 'Down']
    )

st.divider()

# ---------------- PREDICTION ---------------- #

if st.button("Predict Heart Disease Risk"):

    # Create raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'MaxHR': MaxHR,
        'Oldpeak': Oldpeak,

        'Sex_' + sex: 1,
        'ChestPainType_' + pain: 1,
        'RestingECG_' + RestingECG: 1,
        'ExerciseAngina_' + ExerciseAngina: 1,
        'ST_Slope_' + ST_Slope: 1
    }

    # Convert to dataframe
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # Probability
    probability = model.predict_proba(scaled_input)[0][1]

    st.divider()

    # ---------------- RESULT ---------------- #

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ High Risk of Heart Disease\n\n"
            f"Prediction Confidence: {probability * 100:.2f}%"
        )

    else:
        st.success(
            f"✅ Low Risk of Heart Disease\n\n"
            f"Prediction Confidence: {(1 - probability) * 100:.2f}%"
        )

    # ---------------- SHOW USER INPUT ---------------- #

    with st.expander("View Submitted Patient Information"):

        st.dataframe(input_df)

    # ---------------- HEALTH TIPS ---------------- #

    st.divider()

    st.subheader("💡 General Health Tips")

    st.markdown("""
    - Maintain a healthy diet  
    - Exercise regularly  
    - Avoid smoking  
    - Control cholesterol levels  
    - Monitor blood pressure  
    - Get regular medical checkups  
    """)
