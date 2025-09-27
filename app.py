import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import json
import streamlit_lottie as st_lottie
import requests
from fpdf import FPDF

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report

st.set_page_config(page_title="Disease Prediction App", layout="wide")
st.title("\U0001F9E0 AI-Powered Disease Prediction App")
st.write("Predict *Diabetes, Heart Disease, or Breast Cancer* using Machine Learning")

# Background styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);
        color: #333;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
st.sidebar.title("Navigation")
disease = st.sidebar.radio("Select Disease to Predict:", ["Home", "Diabetes", "Heart Disease", "Breast Cancer"])

st.sidebar.markdown("""
### How to Use
1. Choose a disease
2. Fill the form
3. Submit to get prediction

**Disclaimer:** Consult your doctor for actual diagnosis.
""")

st.sidebar.subheader("\U0001F6E0 Debugging Info")
st.sidebar.write("Files in Directory:")
st.sidebar.write(os.listdir())

# Load animation
@st.cache_data(show_spinner=False)
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_tutvdkg0.json")

# Home Page
if disease == "Home":
    col1, col2 = st.columns([1, 2])
    with col1:
        st_lottie.st_lottie(lottie_ai, height=300, key="ai")
    with col2:
        st.subheader("Welcome to Your Health Companion")
        st.write("This app uses machine learning to help you get predictions for diabetes, heart disease, and breast cancer.")
        st.write("Just head over to the sidebar, pick a condition, and fill out the form to get started.")
        st.success("Your health prediction starts here!")

# Load or train models
MODEL_FILES = {
    "diabetes_model": "diabetes_model.pkl",
    "diabetes_scaler": "diabetes_scaler.pkl",
    "heart_model": "heart_model.pkl",
    "heart_scaler": "heart_scaler.pkl",
    "cancer_model": "cancer_model.pkl",
    "cancer_scaler": "cancer_scaler.pkl",
    "cancer_features": "cancer_features.pkl"
}

TRAIN_DATA = {"diabetes": None, "heart": None, "cancer": None}

@st.cache_resource
def load_models():
    diabetes_model = joblib.load(MODEL_FILES["diabetes_model"])
    diabetes_scaler = joblib.load(MODEL_FILES["diabetes_scaler"])
    heart_model = joblib.load(MODEL_FILES["heart_model"])
    heart_scaler = joblib.load(MODEL_FILES["heart_scaler"])
    cancer_model = joblib.load(MODEL_FILES["cancer_model"])
    cancer_scaler = joblib.load(MODEL_FILES["cancer_scaler"])
    cancer_features = joblib.load(MODEL_FILES["cancer_features"])

    diabetes_df = pd.read_csv("diabetes.csv")
    X_d = diabetes_df.drop("Outcome", axis=1)
    y_d = diabetes_df["Outcome"]
    X_d_train, _, y_d_train, _ = train_test_split(X_d, y_d, test_size=0.2, random_state=0)
    X_d_train_scaled = diabetes_scaler.transform(X_d_train)

    heart_df = pd.read_csv("heart.csv")
    X_h = heart_df.drop("target", axis=1)
    y_h = heart_df["target"]
    X_h_train, _, y_h_train, _ = train_test_split(X_h, y_h, test_size=0.2, random_state=0)
    X_h_train_scaled = heart_scaler.transform(X_h_train)

    cancer = load_breast_cancer()
    X_c = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y_c = cancer.target
    X_c_train, _, y_c_train, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=0)
    X_c_train_scaled = cancer_scaler.transform(X_c_train)

    return {
        "diabetes": (diabetes_model, diabetes_scaler),
        "heart": (heart_model, heart_scaler),
        "cancer": (cancer_model, cancer_scaler, cancer_features)
    }, {
        "diabetes": (X_d_train_scaled, y_d_train),
        "heart": (X_h_train_scaled, y_h_train),
        "cancer": (X_c_train_scaled, y_c_train)
    }

models, TRAIN_DATA = load_models()

# Predict with explanation
def predict_with_explanation(model, scaler, input_data, labels):
    X = np.array(input_data).reshape(1, -1)
    X_scaled = scaler.transform(X)
    result = model.predict(X_scaled)[0]
    probas = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
    confidence = probas[result]
    explanation = generate_explanation(result, input_data)
    plot_chances_graph(probas, list(labels.values()), "Chances of Disease")
    return labels[result], confidence, explanation

def generate_explanation(result, input_data):
    return "You may want to consult a professional for further examination." if result else "Your indicators suggest a low risk."

def plot_chances_graph(probabilities, labels, title):
    fig, ax = plt.subplots()
    bars = ax.bar(labels, probabilities, color=['green', 'red'])
    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{prob:.2f}", ha='center', fontsize=10)
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.1)
    st.pyplot(fig)

def download_report_pdf(prediction, confidence, explanation):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=16)
    pdf.cell(200, 10, txt="Health Prediction Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 10, txt=f"Prediction: {prediction}", ln=True)
    pdf.cell(0, 10, txt=f"Confidence: {confidence:.2f}", ln=True)
    pdf.multi_cell(0, 10, txt=f"Explanation: {explanation}")
    pdf.ln(10)
    pdf.cell(0, 10, txt="Date: "+pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'), ln=True)
    pdf_output = pdf.output(dest='S').encode('latin1')
    st.download_button(
        "Download Health Report (PDF)",
        data=pdf_output,
        file_name="health_report.pdf",
        mime="application/pdf"
    )

def model_performance(model, X_train, y_train):
    y_pred = model.predict(X_train)
    return classification_report(y_train, y_pred)

# Forms and prediction logic moved to modular script files (imported separately)
# from diabetes_form import show_diabetes_form
# from heart_form import show_heart_form
# from cancer_form import show_cancer_form

def show_diabetes_form(models):
    st.header("Diabetes Prediction")
    model, scaler = models["diabetes"]
    with st.form("diabetes_form"):
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        submitted = st.form_submit_button("Submit Diabetes Data")
    if submitted:
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        label, confidence, explanation = predict_with_explanation(model, scaler, input_data, {
            0: "🟢 You are unlikely to have diabetes.",
            1: "🔴 The results suggest a possible risk of diabetes. Please consult a healthcare provider."
        })
        st.success(f"Prediction: {label} (Confidence: {confidence:.2f})")
        st.text(model_performance(model, TRAIN_DATA["diabetes"][0], TRAIN_DATA["diabetes"][1]))
        st.text(f"Explanation: {explanation}")
        download_report_pdf(label, confidence, explanation)

def show_heart_form(models):
    st.header("Heart Disease Prediction")
    model, scaler = models["heart"]
    with st.form("heart_form"):
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thal", [0, 1, 2, 3])
        submitted = st.form_submit_button("Submit Heart Data")
    if submitted:
        input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        label, confidence, explanation = predict_with_explanation(model, scaler, input_data, {
            0: "🟢 Your heart health looks good based on this input.",
            1: "🔴 These results show possible signs of heart-related issues. Please consult your doctor."
        })
        st.success(f"Prediction: {label} (Confidence: {confidence:.2f})")
        st.text(model_performance(model, TRAIN_DATA["heart"][0], TRAIN_DATA["heart"][1]))
        st.text(f"Explanation: {explanation}")
        download_report_pdf(label, confidence, explanation)

def show_cancer_form(models):
    st.header("Breast Cancer Prediction")
    st.info("This demo uses only the first 10 features of the breast cancer dataset for prediction. For best results, use all features in a real application.")
    model, scaler, _ = models["cancer"]
    with st.form("cancer_form"):
        mean_radius = st.number_input("Mean Radius", min_value=0.0, max_value=30.0, value=14.0, format="%.2f")
        mean_texture = st.number_input("Mean Texture", min_value=0.0, max_value=40.0, value=20.0, format="%.2f")
        mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, max_value=200.0, value=90.0, format="%.2f")
        mean_area = st.number_input("Mean Area", min_value=0.0, max_value=2500.0, value=600.0, format="%.2f")
        mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, max_value=0.5, value=0.1, format="%.3f")
        mean_compactness = st.number_input("Mean Compactness", min_value=0.0, max_value=1.0, value=0.1, format="%.3f")
        mean_concavity = st.number_input("Mean Concavity", min_value=0.0, max_value=1.0, value=0.1, format="%.3f")
        mean_concave_points = st.number_input("Mean Concave Points", min_value=0.0, max_value=1.0, value=0.05, format="%.3f")
        mean_symmetry = st.number_input("Mean Symmetry", min_value=0.0, max_value=1.0, value=0.2, format="%.3f")
        mean_fractal_dimension = st.number_input("Mean Fractal Dimension", min_value=0.0, max_value=0.2, value=0.06, format="%.3f")
        submitted = st.form_submit_button("Submit Cancer Data")
    if submitted:
        input_data = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension]
        label, confidence, explanation = predict_with_explanation(model, scaler, input_data, {
            0: "🟢 You have a low risk for breast cancer.",
            1: "🔴 The risk of breast cancer seems elevated. Please get a professional consultation."
        })
        st.success(f"Prediction: {label} (Confidence: {confidence:.2f})")
        st.text(model_performance(model, TRAIN_DATA["cancer"][0], TRAIN_DATA["cancer"][1]))
        st.text(f"Explanation: {explanation}")
        download_report_pdf(label, confidence, explanation)

if disease == "Diabetes":
    show_diabetes_form(models)
elif disease == "Heart Disease":
    show_heart_form(models)
elif disease == "Breast Cancer":
    show_cancer_form(models)

# Footer contact form
st.markdown("---")
with st.expander("\U0001F4E7 Contact Us"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")
    if st.button("Send Message"):
        st.success("Thank you for contacting us! We'll get back to you soon.")
