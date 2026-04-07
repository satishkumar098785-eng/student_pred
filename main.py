import streamlit as st
import numpy as np
import pickle
import pandas as pd

# -----------------------------
# Load Models Safely
# -----------------------------
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("placement_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        mapping = pickle.load(open("placement_mapping.pkl", "rb"))
        return model, scaler, mapping
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, scaler, placement_mapping = load_assets()

if model is None:
    st.stop()

placement_labels = {v: k for k, v in placement_mapping.items()}

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.title("🎓 Student Placement Prediction System")
st.markdown(
    "Predict whether a student is likely to be **placed** based on academic performance and skills."
)

st.divider()

# -----------------------------
# Input Section
# -----------------------------
st.subheader("📥 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Academic Scores")
    ssc = st.slider("SSC Percentage", 40, 100, 60)
    hsc = st.slider("HSC Percentage", 40, 100, 60)
    degree = st.slider("Degree Percentage", 40, 100, 60)

with col2:
    st.markdown("### 🧠 Skill Assessment")
    coding = st.slider("Coding Skill", 1, 100, 50)
    communication = st.slider("Communication Skill", 1, 100, 50)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Placement", use_container_width=True):

    input_data = np.array([[ssc, hsc, degree, coding, communication]])
    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    result = placement_labels.get(int(prediction) * 100, "Unknown")
    confidence = probabilities[prediction] * 100

    # -----------------------------
    # Display Results
    # -----------------------------
    st.subheader("📋 Student Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("SSC", f"{ssc}%")
    col2.metric("HSC", f"{hsc}%")
    col3.metric("Degree", f"{degree}%")

    col1, col2 = st.columns(2)
    col1.metric("Coding", coding)
    col2.metric("Communication", communication)

    st.divider()

    # Prediction Output
    st.subheader("🎯 Prediction Result")

    if result == "Placed":
        st.success(f"✅ {result}")
    else:
        st.error(f"❌ {result}")

    st.metric("Confidence Score", f"{confidence:.2f}%")

    st.divider()

    # -----------------------------
    # Probability Chart
    # -----------------------------
    st.subheader("📊 Probability Distribution")

    prob_df = pd.DataFrame({
        "Placement Status": ["Placed", "Not Placed"],
        "Probability": probabilities
    })

    st.bar_chart(prob_df.set_index("Placement Status"))

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Developed with ❤️ using Streamlit")