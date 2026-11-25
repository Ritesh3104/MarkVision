
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("student_data.csv")

# Input features
X = data.drop("Final_Score", axis=1)

# Target
y = data["Final_Score"]

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# Save model
joblib.dump(model, "student_model.pkl")

print("Model trained successfully!")
print("Model saved as student_model.pkl")
