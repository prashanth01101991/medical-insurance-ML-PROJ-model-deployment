import streamlit as st
import joblib
import pandas as pd

# Load Model
model = joblib.load("model.pkl")

st.title("Medical Insurance Cost Predictor")
# Display an image (local file or URL)
st.image("Insu.jpg", caption="PK Health Insurance",  use_container_width =True)

# Option 1: Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file with input data", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(input_data)

    if st.button("Predict"):
        predictions = model.predict(input_data)
        # Add predictions as a new column
        input_data['Predicted_Expenses'] = predictions
        st.write("Predictions:")
        st.dataframe(input_data)

        # Optionally, allow download of results
        csv = input_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="predicted_expenses.csv",
            mime="text/csv"
        )

# Option 2: Keep single input form if no CSV is uploaded
else:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
    sex = st.radio("Sex", ["male", "female"])
    smoker = st.radio("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    input_data = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region],
        }
    )

    if st.button("Predict Single"):
        prediction = model.predict(input_data)
        st.success(f"Estimated Insurance Cost: ${prediction[0]:.2f}")
