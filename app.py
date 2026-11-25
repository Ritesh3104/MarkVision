
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="MarkVision",
    page_icon="🎓",
    layout="centered"
)

# CSS Design
st.markdown("""
<style>
.stApp {
    background: #f5f6fc;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
}

.header {
    background: linear-gradient(110deg, #5145d8, #8876ef);
    padding: 30px;
    border-radius: 14px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
}

.header h1 {
    color: white;
    font-size: 35px;
    margin-bottom: 8px;
}

.header p {
    color: #eeeeff;
    font-size: 15px;
}

h2, h3 {
    color: #25254b;
}

.stButton > button {
    background: #6254dc;
    color: white;
    border-radius: 8px;
    height: 48px;
    width: 100%;
    border: none;
    font-size: 16px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #493bc0;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("student_model.pkl")

# Header
st.markdown("""
<div class="header">
    <h1>MarkVision</h1>
    <p>Student Performance Prediction System</p>
</div>
""", unsafe_allow_html=True)

st.subheader("Enter Student Details")
st.write("Enter the following information to predict the student's expected marks.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input(
        "Study Hours Per Day",
        0.0, 16.0, 4.0, 0.5
    )

    attendance = st.number_input(
        "Attendance (%)",
        0, 100, 75
    )

    previous_score = st.number_input(
        "Previous Exam Score",
        0, 100, 65
    )

with col2:
    sleep_hours = st.number_input(
        "Sleep Hours Per Day",
        0.0, 16.0, 7.0, 0.5
    )

    assignment_average = st.number_input(
        "Assignment Average (%)",
        0, 100, 70
    )

    practice_tests = st.number_input(
        "Practice Tests Per Month",
        0, 30, 2
    )

st.write("")

# Prediction
if st.button("Predict Student Score"):

    input_data = pd.DataFrame([{
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Score": previous_score,
        "Sleep_Hours": sleep_hours,
        "Assignment_Average": assignment_average,
        "Practice_Tests": practice_tests
    }])

    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))

    st.markdown("---")
    st.subheader("Prediction Result")

    st.markdown(
        f"<h1 style='color:#594bd6;text-align:center;'>"
        f"{prediction:.1f} / 100</h1>",
        unsafe_allow_html=True
    )

    if prediction >= 75:
        st.success("Excellent Performance!")
    elif prediction >= 50:
        st.info("Good Performance!")
    else:
        st.warning("Needs Improvement")

    st.caption(
        "Predicted marks are estimates based on the trained model."
    )
