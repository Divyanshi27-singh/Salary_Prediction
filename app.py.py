import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("salary_predictor.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("💼 Salary Prediction App")

# Input fields
work_year = st.selectbox("Work Year", [2020, 2021, 2022, 2023, 2024, 2025])
experience_level = st.selectbox("Experience Level", ['EN', 'MI', 'SE', 'EX'])
employment_type = st.selectbox("Employment Type", ['FT', 'PT', 'CT', 'FL'])
job_title = st.text_input("Job Title", "Data Scientist")
employee_residence = st.text_input("Employee Residence (e.g., IN, US, UK)", "IN")
remote_ratio = st.slider("Remote Ratio (%)", 0, 100, 0)
company_location = st.text_input("Company Location (e.g., IN, US, UK)", "IN")
company_size = st.selectbox("Company Size", ['S', 'M', 'L'])

# Create DataFrame
input_df = pd.DataFrame([{
    'work_year': work_year,
    'experience_level': experience_level,
    'employment_type': employment_type,
    'job_title': job_title,
    'employee_residence': employee_residence,
    'remote_ratio': remote_ratio,
    'company_location': company_location,
    'company_size': company_size
}])

# Apply label encoding to categorical columns
for col in ['experience_level', 'employment_type', 'job_title', 
            'employee_residence', 'company_location', 'company_size']:
    if col in label_encoder:
        input_df[col] = label_encoder[col].transform(input_df[col])

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Salary in USD: ${prediction[0]:,.2f}")