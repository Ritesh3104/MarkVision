# MarkVision – Student Performance Prediction

## Project Overview
MarkVision is a Machine Learning-based web application that predicts student performance based on academic and lifestyle-related parameters.

The application uses Python, Random Forest Regression and Streamlit to estimate students' expected marks.

## Features
- Student marks prediction
- Study hours analysis
- Attendance analysis
- Previous exam score analysis
- Sleep hours analysis
- Assignment performance analysis
- Practice test analysis
- Interactive prediction interface
- Data visualization and model evaluation

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook
- Plotly

## Dataset Features
1. Study Hours
2. Attendance
3. Previous Score
4. Sleep Hours
5. Assignment Average
6. Practice Tests
7. Final Score (Target Variable)

**Note:** The current dataset contains 20 illustrative sample records for educational demonstration.

## Exploratory Data Analysis
The Jupyter Notebook includes:
- Dataset overview
- Missing value analysis
- Duplicate record checking
- Statistical analysis
- Study Hours vs Final Score
- Attendance vs Final Score
- Student Score Distribution
- Correlation Heatmap

## Machine Learning
**Algorithm:** Random Forest Regressor

Model evaluation:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Project Structure
```text
MarkVision/
├── app.py
├── train_model.py
├── student_data.csv
├── student_data_cleaned.csv
├── student_model.pkl
├── requirements.txt
├── MarkVision_Analysis.ipynb
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python train_model.py
```

## Run Application

```bash
python -m streamlit run app.py
```

## Disclaimer
Predictions are estimates based on an illustrative sample dataset and are not validated for real-world academic assessment.