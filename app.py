import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#edf6ff,#ffffff);
}

h1{
    font-size:48px;
    font-weight:900;
    letter-spacing:1px;
} 

.block-container{
    padding-top:2rem;
}

.stButton>button{

background:linear-gradient(90deg,#D62828,#E63946);

height:60px;

font-size:22px;

font-weight:bold;

border-radius:15px;

transition:0.3s;

border:none;

color:white;

}

.stButton>button:hover{

transform:scale(1.03);

box-shadow:0px 10px 20px rgba(214,40,40,.4);

}


div[data-testid="stMetric"]{
    background:white;
    border-radius:15px;
    padding:15px;
    box-shadow:0 5px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")
# -----------------------------
# HEADING
# -----------------------------

st.markdown(
"""
<h1 style='text-align:center;'>
❤️ Heart Disease Prediction
</h1>

<p style='text-align:center;
font-size:20px;
color:gray;'>

Machine Learning Based Prediction System

</p>
""",
unsafe_allow_html=True
)

st.write("")

# -----------------------------
# TOP INFO CARDS
# -----------------------------

st.markdown("""
<div style="
background: linear-gradient(135deg,#D62828,#E63946);
padding:35px;
border-radius:20px;
color:white;
text-align:center;
box-shadow:0px 10px 25px rgba(0,0,0,0.2);
margin-bottom:30px;
">

<h1 style="
font-size:48px;
margin-bottom:10px;
font-weight:900;
color:white;">

❤️ Heart Disease Prediction

</h1>

<p style="
font-size:20px;
margin-top:10px;
">

Predict the likelihood of heart disease using Machine Learning.
<br>
Enter the patient's details below and get an instant prediction.

</p>

</div>
""", unsafe_allow_html=True)
    # -----------------------------
# PATIENT DETAILS
# -----------------------------

st.write("")
st.markdown("""
<h2 style='color:#D62828;'>
🩺 Patient Information
</h2>
""", unsafe_allow_html=True)

left, right = st.columns(2)

with left:

    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=40
    )

    sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    bp = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=220,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=0,
        max_value=700,
        value=200
    )

with right:

    fasting = st.selectbox(
        "Fasting Blood Sugar",
        ["No", "Yes"]
    )

    ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    hr = st.slider(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150
    )

    oldpeak = st.slider(
        "Old Peak",
        min_value=0.0,
        max_value=6.5,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

    angina = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )

st.write("")

predict = st.button("❤️ Predict Heart Disease")
# -----------------------------
# FEATURE ENCODING
# -----------------------------

if predict:

    data = {}

    # Initialize all features to 0
    for col in columns:
        data[col] = 0

    # Numerical Features
    data["Age"] = age
    data["RestingBP"] = bp
    data["Cholesterol"] = chol
    data["FastingBS"] = 1 if fasting == "Yes" else 0
    data["MaxHR"] = hr
    data["Oldpeak"] = oldpeak

    # Gender
    if sex == "Male":
        data["Sex_M"] = 1

    # Chest Pain
    if cp == "ATA":
        data["ChestPainType_ATA"] = 1

    elif cp == "NAP":
        data["ChestPainType_NAP"] = 1

    elif cp == "TA":
        data["ChestPainType_TA"] = 1

    # ASY remains all zeros

    # ECG
    if ecg == "Normal":
        data["RestingECG_Normal"] = 1

    elif ecg == "ST":
        data["RestingECG_ST"] = 1

    # LVH remains all zeros

    # Exercise Angina
    if angina == "Yes":
        data["ExerciseAngina_Y"] = 1

    # ST Slope
    if slope == "Flat":
        data["ST_Slope_Flat"] = 1

    elif slope == "Up":
        data["ST_Slope_Up"] = 1

    # Down remains all zeros

    # -----------------------------
    # DATAFRAME
    # -----------------------------

    input_df = pd.DataFrame([data])

    input_df = input_df[columns]

    # -----------------------------
    # SCALE
    # -----------------------------

    input_scaled = scaler.transform(input_df)

    # -----------------------------
    # PREDICTION
    # -----------------------------

    prediction = model.predict(input_scaled)[0]

    probability = None

    if hasattr(model, "predict_proba"):
        probability = np.max(model.predict_proba(input_scaled)) * 100

    st.write("")
    st.markdown("---")
    st.subheader("🩺 Prediction Result")

    if prediction == 0:

        st.markdown("""
        <div style="
        background:#d4edda;
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:#155724;
        font-size:30px;
        font-weight:bold;">
        🟢 LOW RISK
        </div>
        """, unsafe_allow_html=True)

        st.success("No significant signs of heart disease were detected.")

        st.info("""
### ❤️ Recommendations

- Exercise regularly
- Eat a balanced diet
- Maintain healthy weight
- Sleep 7–8 hours
- Annual health check-up
""")

    else:

        st.markdown("""
        <div style="
        background:#f8d7da;
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:#721c24;
        font-size:30px;
        font-weight:bold;">
        🔴 HIGH RISK
        </div>
        """, unsafe_allow_html=True)

        st.error("The model predicts a higher likelihood of heart disease.")

        st.warning("""
### 🩺 Recommendations

- Consult a Cardiologist
- Monitor Blood Pressure
- Reduce Cholesterol
- Avoid Smoking
- Exercise under medical supervision
""")

    if probability is not None:

        st.write("")

        st.metric(
            "Prediction Confidence",
            f"{probability:.2f}%"
        )
        st.markdown("---")

    st.subheader("📋 Patient Summary")

    summary = pd.DataFrame({

        "Feature":[
            "Age",
            "Gender",
            "Chest Pain",
            "Blood Pressure",
            "Cholesterol",
            "Fasting Sugar",
            "ECG",
            "Max Heart Rate",
            "Exercise Angina",
            "Old Peak",
            "ST Slope"
        ],

        "Value":[
            age,
            sex,
            cp,
            bp,
            chol,
            fasting,
            ecg,
            hr,
            angina,
            oldpeak,
            slope
        ]

    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")
st.caption("❤️ Developed by Vaishnavi Kamboj")