import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('salary_voting_model.pkl')

st.title("Salary Prediction App")

# --- User Inputs ---
work_year = st.number_input('Work Year', min_value=2018, max_value=2025, value=2021)

experience_level = st.selectbox('Experience Level', options=['EN', 'MI', 'SE', 'EX'])
company_size = st.selectbox('Company Size', options=['S', 'M', 'L'])
job_title = st.text_input('Job Title (Enter code or integer)')

remote_ratio = st.slider('Remote Ratio (%)', 0, 100, 0, step=25)

employment_type = st.selectbox('Employment Type', options=['FT', 'PT', 'CT', 'FL'])
employee_residence = st.selectbox('Employee Residence', options=['US', 'IN', 'GB', 'DE', 'CA', 'BR', 'FR', 'NL', 'JP', 'IT', 'AU', 'CA', 'PL', 'ES', 'DK', 'MX', 'RU', 'AT', 'BE', 'CH', 'HK', 'CN', 'PT', 'CO', 'IE', 'IL', 'SG', 'SE', 'HU', 'RO', 'CZ', 'NG', 'PK', 'TR', 'LU', 'IR', 'UA', 'GR', 'KE', 'CL', 'VN', 'AR', 'NO', 'EC', 'PE', 'ZA', 'MD', 'MT', 'AS', 'CR', 'BG', 'SI'])
company_location = st.selectbox('Company Location', options=['US', 'IN', 'GB', 'DE', 'CA', 'BR', 'FR', 'NL', 'JP', 'IT', 'AU', 'CA', 'PL', 'ES', 'DK', 'MX', 'RU', 'AT', 'BE', 'CH', 'HK', 'CN', 'PT', 'CO', 'IE', 'IL', 'SG', 'SE', 'HU', 'RO', 'CZ', 'NG', 'PK', 'TR', 'LU', 'IR', 'UA', 'GR', 'KE', 'CL', 'VN', 'AR', 'NO', 'EC', 'PE', 'ZA', 'MD', 'MT', 'AS', 'CR', 'BG', 'SI'])

# --- Encoding mappings (Adjust to your training encoding) ---
experience_level_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
company_size_map = {'S': 0, 'M': 1, 'L': 2}

# Encode inputs
exp_lvl = experience_level_map.get(experience_level, 0)
comp_size = company_size_map.get(company_size, 0)

# Encode job_title (expecting integer code; default 0 if invalid)
try:
    job_title_encoded = int(job_title)
except:
    job_title_encoded = 0

# --- Prepare input dataframe ---
# Create a dataframe with all features set to 0 by default
input_data = pd.DataFrame(columns=[
    'work_year', 'experience_level', 'job_title', 'remote_ratio', 'company_size',
    'employment_type_FL', 'employment_type_FT', 'employment_type_PT',
    'employee_residence_AT', 'employee_residence_BE', 'employee_residence_BG',
    'employee_residence_BR', 'employee_residence_CA', 'employee_residence_CL',
    'employee_residence_CN', 'employee_residence_CO', 'employee_residence_DE',
    'employee_residence_DK', 'employee_residence_ES', 'employee_residence_FR',
    'employee_residence_GB', 'employee_residence_GR', 'employee_residence_HK',
    'employee_residence_HR', 'employee_residence_HU', 'employee_residence_IN',
    'employee_residence_IR', 'employee_residence_IT', 'employee_residence_JE',
    'employee_residence_JP', 'employee_residence_KE', 'employee_residence_LU',
    'employee_residence_MD', 'employee_residence_MT', 'employee_residence_MX',
    'employee_residence_NG', 'employee_residence_NL', 'employee_residence_NZ',
    'employee_residence_PH', 'employee_residence_PK', 'employee_residence_PL',
    'employee_residence_PR', 'employee_residence_PT', 'employee_residence_RO',
    'employee_residence_RS', 'employee_residence_RU', 'employee_residence_SG',
    'employee_residence_SI', 'employee_residence_TR', 'employee_residence_UA',
    'employee_residence_US', 'employee_residence_VN',
    'company_location_AS', 'company_location_AT', 'company_location_BE',
    'company_location_BR', 'company_location_CA', 'company_location_CH',
    'company_location_CL', 'company_location_CN', 'company_location_CO',
    'company_location_DE', 'company_location_DK', 'company_location_ES',
    'company_location_FR', 'company_location_GB', 'company_location_GR',
    'company_location_HR', 'company_location_HU', 'company_location_IL',
    'company_location_IN', 'company_location_IR', 'company_location_IT',
    'company_location_JP', 'company_location_KE', 'company_location_LU',
    'company_location_MD', 'company_location_MT', 'company_location_MX',
    'company_location_NG', 'company_location_NL', 'company_location_NZ',
    'company_location_PK', 'company_location_PL', 'company_location_PT',
    'company_location_RU', 'company_location_SG', 'company_location_SI',
    'company_location_TR', 'company_location_UA', 'company_location_US',
    'company_location_VN'
])

# Initialize all values to 0
input_data.loc[0] = 0

# Set the values for the features that are not one-hot encoded
input_data.loc[0, 'work_year'] = work_year
input_data.loc[0, 'experience_level'] = exp_lvl
input_data.loc[0, 'job_title'] = job_title_encoded
input_data.loc[0, 'remote_ratio'] = remote_ratio
input_data.loc[0, 'company_size'] = comp_size

# Set the one-hot encoded columns based on user input
# Employment type
if employment_type == 'FL':
    input_data.loc[0, 'employment_type_FL'] = 1
elif employment_type == 'FT':
    input_data.loc[0, 'employment_type_FT'] = 1
elif employment_type == 'PT':
    input_data.loc[0, 'employment_type_PT'] = 1

# Employee residence
residence_col = f'employee_residence_{employee_residence}'
if residence_col in input_data.columns:
    input_data.loc[0, residence_col] = 1

# Company location
location_col = f'company_location_{company_location}'
if location_col in input_data.columns:
    input_data.loc[0, location_col] = 1

# --- Predict button ---
if st.button('Predict Salary'):
    prediction = model.predict(input_data)
    st.success(f'Predicted Salary in USD: ${prediction[0]:,.2f}')
