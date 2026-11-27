from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(
    page_title="MarkVision — Student Performance Studio",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Soft Faint Background & Clean Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #f4f7fc;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1160px;
    }

    [data-testid="stSidebar"] {
        background-color: #f9fbff;
        border-right: 1px solid #e2e8f2;
    }

    .mv-hero {
        background: linear-gradient(90deg, #5643cc 0%, #6b5ce7 55%, #7d6ef0 100%);
        border-radius: 14px;
        padding: 1.75rem 2rem;
        text-align: center;
        color: #ffffff;
        margin-bottom: 1.75rem;
        box-shadow: 0 4px 14px rgba(86, 67, 204, 0.16);
    }
    .mv-hero h1 {
        color: #ffffff !important;
        font-size: 2.25rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.02em;
    }
    .mv-hero p {
        color: #edeaff;
        font-size: 0.98rem;
        margin: 0;
    }

    .mv-dash-banner {
        background: linear-gradient(90deg, #eef3ff 0%, #f0f8ff 55%, #f5f1ff 100%);
        border: 1px solid #dce6f7;
        border-radius: 14px;
        padding: 1.35rem 1.75rem;
        margin-bottom: 1.3rem;
    }
    .mv-dash-banner h1 {
        color: #1e293b !important;
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
    }
    .mv-dash-banner p {
        color: #526074;
        font-size: 0.95rem;
        margin: 0;
    }

    .mv-kpi {
        background-color: #ffffff;
        border: 1px solid #e3e9f2;
        border-radius: 13px;
        padding: 16px 18px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.025);
    }
    .mv-kpi-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .mv-kpi-num {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
    }
    .mv-kpi-note {
        font-size: 0.78rem;
        color: #8492a6;
        margin-top: 4px;
    }

    .mv-score-big {
        text-align: center;
        font-size: 2.75rem;
        font-weight: 800;
        color: #5643cc;
        margin: 0.75rem 0 1rem 0;
        line-height: 1.1;
    }

    .mv-status-banner {
        background-color: #e8f1fc;
        color: #1e4e8c;
        border-radius: 8px;
        padding: 0.95rem 1.15rem;
        font-size: 0.98rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
    }

    .mv-caption {
        color: #6c757d;
        font-size: 0.85rem;
    }

    div.stButton > button {
        background-color: #6b5ce7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.4rem !important;
        font-weight: 600 !important;
    }
    div.stButton > button:hover {
        background-color: #5643cc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Sleep_Hours",
    "Assignment_Average",
    "Practice_Tests",
]
TARGET = "Final_Score"


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    csv_path = Path("student_data.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    rng = np.random.default_rng(42)
    n = 1000
    study_hours = np.round(np.clip(rng.normal(4.2, 1.6, n), 0.5, 10.0), 2)
    attendance = np.round(np.clip(rng.normal(78.0, 11.0, n), 45.0, 100.0), 0)
    previous_score = np.round(np.clip(rng.normal(68.0, 14.0, n), 30.0, 100.0), 0)
    sleep_hours = np.round(np.clip(rng.normal(7.0, 1.1, n), 4.0, 10.0), 2)
    assignment_avg = np.round(np.clip(rng.normal(73.0, 12.0, n), 35.0, 100.0), 0)
    practice_tests = rng.integers(0, 10, size=n)

    noise = rng.normal(0, 1.8, n)
    final_score = (
        6.1
        + 3.20 * study_hours
        + 0.18 * attendance
        + 0.25 * previous_score
        + 0.90 * sleep_hours
        + 0.12 * assignment_avg
        + 0.85 * practice_tests
        + noise
    )
    final_score = np.round(np.clip(final_score, 30.0, 100.0), 1)

    return pd.DataFrame({
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Score": previous_score,
        "Sleep_Hours": sleep_hours,
        "Assignment_Average": assignment_avg,
        "Practice_Tests": practice_tests,
        "Final_Score": final_score,
    })


@st.cache_resource
def load_model(df: pd.DataFrame):
    model_path = Path("student_model.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    return LinearRegression().fit(df[FEATURES], df[TARGET])


def classify_score_band(score: float) -> str:
    if score < 50:
        return "Below 50"
    if score <= 70:
        return "50–70"
    if score <= 85:
        return "70–85"
    return "85–100"


df = load_dashboard_data()
model = load_model(df)

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 MarkVision")
    st.caption("STUDENT PERFORMANCE STUDIO")
    st.divider()
    page = st.radio(
        "Select Page",
        options=["🎯 Score Predictor", "📊 Dashboard Overview"],
    )
    st.divider()
    st.caption(f"• Records: {len(df):,}")

# ---------------------------------------------------------------------------
# Page 1: Score Predictor
# ---------------------------------------------------------------------------
if page == "🎯 Score Predictor":
    st.markdown(
        """
        <div class="mv-hero">
            <h1>MarkVision</h1>
            <p>Student Performance Prediction System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Enter Student Details")
    st.markdown("Enter the following information to predict the student's expected marks.")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        study_hours = st.number_input("Study Hours Per Day", 0.0, 24.0, 4.00, 0.50, format="%.2f")
        attendance = st.number_input("Attendance (%)", 0, 100, 75, 1)
        previous_score = st.number_input("Previous Exam Score", 0, 100, 65, 1)

    with col_right:
        sleep_hours = st.number_input("Sleep Hours Per Day", 0.0, 24.0, 7.00, 0.50, format="%.2f")
        assignment_avg = st.number_input("Assignment Average (%)", 0, 100, 70, 1)
        practice_tests = st.number_input("Practice Tests Per Month", 0, 30, 2, 1)

    predict_btn = st.button("Predict Student Score")

    st.divider()

    if "predicted_score" not in st.session_state:
        st.session_state.predicted_score = 65.1

    if predict_btn:
        input_df = pd.DataFrame(
            [[study_hours, attendance, previous_score, sleep_hours, assignment_avg, practice_tests]],
            columns=FEATURES,
        )
        st.session_state.predicted_score = round(float(np.clip(model.predict(input_df)[0], 0.0, 100.0)), 1)

    score = st.session_state.predicted_score

    st.markdown("## Prediction Result")
    st.markdown(f'<div class="mv-score-big">{score:.1f} / 100</div>', unsafe_allow_html=True)

    if score >= 80:
        status_text = "Excellent Performance!"
    elif score >= 60:
        status_text = "Good Performance!"
    elif score >= 45:
        status_text = "Average Performance!"
    else:
        status_text = "Needs Improvement!"

    st.markdown(f'<div class="mv-status-banner">{status_text}</div>', unsafe_allow_html=True)
    st.markdown('<div class="mv-caption">Predicted marks are estimates based on the trained model.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page 2: Dashboard Overview
# ---------------------------------------------------------------------------
else:
    st.markdown(
        """
        <div class="mv-dash-banner">
            <h1>MarkVision — Student Performance Overview</h1>
            <p>Interactive student KPIs and soft pastel visualizations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    f1, f2 = st.columns([2, 1.5])
    with f1:
        min_att = st.slider(
            "Minimum Attendance (%)",
            min_value=int(df["Attendance"].min()),
            max_value=int(df["Attendance"].max()),
            value=int(df["Attendance"].min()),
            step=1,
        )
    with f2:
        band_filter = st.selectbox(
            "Score Band",
            options=["All Scores", "Below 50", "50–70", "70–85", "85–100"],
        )

    filtered_df = df[df["Attendance"] >= min_att].copy()
    filtered_df["Score_Band"] = filtered_df[TARGET].apply(classify_score_band)
    if band_filter != "All Scores":
        filtered_df = filtered_df[filtered_df["Score_Band"] == band_filter]

    total_students = int(len(filtered_df))
    avg_score = float(filtered_df[TARGET].mean()) if total_students else 0.0
    avg_attendance = float(filtered_df["Attendance"].mean()) if total_students else 0.0
    avg_study = float(filtered_df["Study_Hours"].mean()) if total_students else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f'<div class="mv-kpi"><div class="mv-kpi-title">Total Students</div><div class="mv-kpi-num">{total_students:,}</div><div class="mv-kpi-note">Attendance ≥ {min_att}%</div></div>',
        unsafe_allow_html=True,
    )
    k2.markdown(
        f'<div class="mv-kpi"><div class="mv-kpi-title">Average Marks</div><div class="mv-kpi-num">{avg_score:.1f} / 100</div><div class="mv-kpi-note">Final score mean</div></div>',
        unsafe_allow_html=True,
    )
    k3.markdown(
        f'<div class="mv-kpi"><div class="mv-kpi-title">Mean Attendance</div><div class="mv-kpi-num">{avg_attendance:.1f}%</div><div class="mv-kpi-note">Class presence</div></div>',
        unsafe_allow_html=True,
    )
    k4.markdown(
        f'<div class="mv-kpi"><div class="mv-kpi-title">Avg Study Time</div><div class="mv-kpi-num">{avg_study:.1f} hrs</div><div class="mv-kpi-note">Daily study hours</div></div>',
        unsafe_allow_html=True,
    )

    st.write("")

    if total_students == 0:
        st.warning("No students match the selected filters.")
    else:
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1.2, 1.4, 1.0])
        with c_ctrl1:
            hist_bins = st.slider("Distribution Bins", min_value=10, max_value=40, value=20, step=2)
        with c_ctrl2:
            scatter_x = st.selectbox("X-Axis Feature", options=FEATURES, index=0)
        with c_ctrl3:
            st.write("")
            show_trend = st.checkbox("Trendline", value=True)

        screen_band_map = {
            "Below 50": "#dfb8d0",
            "50–70": "#c5a8f0",
            "70–85": "#a8d0f8",
            "85–100": "#5cf2c2",
        }
        screen_bar_colors = ["#5cf2c2", "#a8d0f8", "#4aa8e0", "#dfb8d0", "#c27be8", "#c5a8f0", "#7465e2"]

        ch_left, ch_right = st.columns(2)
        with ch_left:
            fig_hist = px.histogram(
                filtered_df,
                x=TARGET,
                color="Score_Band",
                nbins=hist_bins,
                title="Final Marks Distribution",
                color_discrete_map=screen_band_map,
                category_orders={"Score_Band": ["Below 50", "50–70", "70–85", "85–100"]},
                labels={TARGET: "Final Marks", "count": "Students", "Score_Band": "Score Band"},
            )
            fig_hist.update_layout(
                height=340,
                paper_bgcolor="#ffffff",
                plot_bgcolor="#fcfdfe",
                margin=dict(l=15, r=15, t=42, b=15),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with ch_right:
            fig_scat = px.scatter(
                filtered_df,
                x=scatter_x,
                y=TARGET,
                color="Score_Band",
                hover_data=["Study_Hours", "Sleep_Hours", "Attendance"],
                title=f"{scatter_x} vs. Final Marks",
                color_discrete_map=screen_band_map,
                category_orders={"Score_Band": ["Below 50", "50–70", "70–85", "85–100"]},
                labels={scatter_x: scatter_x.replace("_", " "), TARGET: "Final Marks", "Score_Band": "Score Band"},
            )
            if show_trend and total_students >= 2:
                x_vals = filtered_df[scatter_x].to_numpy(dtype=float)
                y_vals = filtered_df[TARGET].to_numpy(dtype=float)
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                fig_scat.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=slope * x_line + intercept,
                        mode="lines",
                        name="Trendline",
                        line=dict(color="#7465e2", width=2.5, dash="dot"),
                    )
                )
            fig_scat.update_layout(
                height=340,
                paper_bgcolor="#ffffff",
                plot_bgcolor="#fcfdfe",
                margin=dict(l=15, r=15, t=42, b=15),
            )
            st.plotly_chart(fig_scat, use_container_width=True)

        b_left, b_right = st.columns(2)
        with b_left:
            labels = ["Below 50", "50–70", "70–85", "85–100"]
            counts = filtered_df["Score_Band"].value_counts().reindex(labels, fill_value=0)
            fig_bar = go.Figure(
                data=[
                    go.Bar(
                        x=counts.index,
                        y=counts.values,
                        text=counts.values,
                        textposition="outside",
                        marker_color=["#dfb8d0", "#c5a8f0", "#a8d0f8", "#5cf2c2"],
                    )
                ]
            )
            fig_bar.update_layout(
                title="Students by Marks Range",
                xaxis_title="Marks Range",
                yaxis_title="Students",
                height=330,
                paper_bgcolor="#ffffff",
                plot_bgcolor="#fcfdfe",
                margin=dict(l=15, r=15, t=42, b=15),
                yaxis=dict(range=[0, max(counts.values) * 1.18 + 5]),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with b_right:
            if total_students >= 2:
                corrs = filtered_df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
                fig_corr = go.Figure(
                    data=[
                        go.Bar(
                            x=corrs.values,
                            y=corrs.index,
                            orientation="h",
                            text=[f"{v:.2f}" for v in corrs.values],
                            textposition="outside",
                            marker_color=screen_bar_colors[: len(corrs)],
                        )
                    ]
                )
                fig_corr.update_layout(
                    title="Feature Correlation with Final Marks",
                    xaxis_title="Correlation (r)",
                    yaxis_title="Feature",
                    height=330,
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#fcfdfe",
                    margin=dict(l=15, r=25, t=42, b=15),
                    xaxis=dict(range=[0, max(0.1, float(corrs.values.max())) + 0.14]),
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Select at least 2 students to compute feature correlations.")