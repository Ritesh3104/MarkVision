import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Sleep_Hours",
    "Assignment_Average",
    "Practice_Tests",
]
TARGET = "Final_Score"


def create_data_and_model(
    csv_path: str = "student_data.csv",
    model_path: str = "student_model.pkl",
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
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

    df = pd.DataFrame({
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Score": previous_score,
        "Sleep_Hours": sleep_hours,
        "Assignment_Average": assignment_avg,
        "Practice_Tests": practice_tests,
        "Final_Score": final_score,
    })
    df.to_csv(csv_path, index=False)

    model = LinearRegression()
    model.fit(df[FEATURES], df[TARGET])
    joblib.dump(model, model_path)
    return df, model


if __name__ == "__main__":
    df, model = create_data_and_model()
    sample = pd.DataFrame([[4.00, 75, 65, 7.00, 70, 2]], columns=FEATURES)
    pred = round(float(model.predict(sample)[0]), 1)
    print(f"Model trained successfully! Sample Prediction: {pred} / 100")