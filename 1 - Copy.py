import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import json
import requests
from streamlit_lottie import st_lottie

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
        st_lottie(lottie_ai, height=300, key="ai")
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
    ax.bar(labels, probabilities, color=['green', 'red'])
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

def download_report(prediction, confidence, explanation):
    result_data = {
        "Prediction": prediction,
        "Confidence": confidence,
        "Explanation": explanation,
    }
    json_report = json.dumps(result_data)
    st.download_button("Download Health Report", json_report, file_name="health_report.json", mime="application/json")

def model_performance(model, X_train, y_train):
    y_pred = model.predict(X_train)
    return classification_report(y_train, y_pred)

# Forms for each disease (diabetes_form, heart_form, cancer_form) here... (unchanged)
# Code for displaying prediction pages (same as your original logic)

# Footer contact form
st.markdown("---")
with st.expander("\U0001F4E7 Contact Us"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")
    if st.button("Send Message"):
        st.success("Thank you for contacting us! We'll get back to you soon.")
