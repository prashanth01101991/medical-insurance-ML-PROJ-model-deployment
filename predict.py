import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")
print("Model loaded successfully!")

# Example: New data to predict
# Make sure column names match exactly with training data
new_data = pd.DataFrame({
    'age': [19, 45],
    'bmi': [27.9, 33.2],
    'children': [0, 2],
    'sex': ['female', 'female'],
    'region':['southwest','northwest'],
    'smoker': ['yes', 'no']
})

# Predict expenses
predictions = model.predict(new_data)

# Display predictions
for i, pred in enumerate(predictions):
    print(f"Predicted expense for record {i+1}: {pred:.2f}")
