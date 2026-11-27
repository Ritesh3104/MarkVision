# MarkVision – Student Performance Prediction

## Project Overview
MarkVision is a machine learning web application built with Python, Scikit-learn, and Streamlit that estimates a student's expected final score from academic and study habits.

## Features
- Student final score prediction
- Study hours and attendance analysis
- Previous score, sleep hours, assignment average, and practice test inputs
- Interactive Streamlit prediction interface
- Exploratory data analysis and model evaluation in Jupyter Notebook

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Dataset Features
- `Study_Hours`
- `Attendance`
- `Previous_Score`
- `Sleep_Hours`
- `Assignment_Average`
- `Practice_Tests`
- `Final_Score` (Target Variable)

## Exploratory Data Analysis
`MarkVision_Analysis.ipynb` covers:
- Dataset and statistical overview
- Study Hours vs Final Score and Attendance vs Final Score scatter plots
- Student Final Score distribution
- Correlation heatmap across features and target
- Actual vs Predicted marks and normalized feature weights

## Machine Learning
Trained using `train_model.py` (`LinearRegression`) and saved to `student_model.pkl`, evaluated with:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Project Structure
```text
MarkVision/
├── app.py
├── train_model.py
├── student_data.csv
├── student_model.pkl
├── requirements.txt
├── .gitignore
├── MarkVision_Analysis.ipynb
└── README.md