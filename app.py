import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load your trained model
model = joblib.load('salary_voting_model.pkl')

# Read the feature columns that the model expects
with open('feature_columns.txt', 'r') as f:
    expected_features = [line.strip() for line in f.readlines()]

# Page setup
st.set_page_config(page_title="Salary Prediction App", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        background: -webkit-linear-gradient(#4CAF50, #2E8B57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #e8f5e9;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #2e7d32;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>💼 Salary Prediction App</h1>", unsafe_allow_html=True)
st.write("Fill in job details below and predict your salary in USD 🚀")

# Sidebar for inputs
st.sidebar.header("Enter Job Details")

work_year = st.sidebar.number_input('📅 Work Year', min_value=2018, max_value=2025, value=2021)
experience_level = st.sidebar.selectbox('🎓 Experience Level', options=['EN', 'MI', 'SE', 'EX'])
company_size = st.sidebar.selectbox('🏢 Company Size', options=['S', 'M', 'L'])
job_title = st.sidebar.text_input('💻 Job Title (Enter code or integer)', value='0')
remote_ratio = st.sidebar.slider('🌍 Remote Ratio (%)', 0, 100, 0, step=25)
employment_type = st.sidebar.selectbox('📋 Employment Type', options=['FT', 'PT', 'CT', 'FL'])

# Get all unique countries from expected features
employee_residence_options = sorted(list(set([f.split('_')[-1] for f in expected_features if f.startswith('employee_residence_')])))
company_location_options = sorted(list(set([f.split('_')[-1] for f in expected_features if f.startswith('company_location_')])))

employee_residence = st.sidebar.selectbox('🏠 Employee Residence', options=employee_residence_options)
company_location = st.sidebar.selectbox('📍 Company Location', options=company_location_options)

# --- Encoding mappings ---
experience_level_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
company_size_map = {'S': 0, 'M': 1, 'L': 2}

# Encode inputs
exp_lvl = experience_level_map.get(experience_level, 0)
comp_size = company_size_map.get(company_size, 0)

# Encode job_title
try:
    job_title_encoded = int(job_title)
except:
    job_title_encoded = 0

# --- Prepare input dataframe with all expected features ---
input_data = pd.DataFrame(columns=expected_features)
input_data.loc[0] = 0  # Initialize all features to 0

# Set the values for non-one-hot features
input_data.loc[0, 'work_year'] = work_year
input_data.loc[0, 'experience_level'] = exp_lvl
input_data.loc[0, 'job_title'] = job_title_encoded
input_data.loc[0, 'remote_ratio'] = remote_ratio
input_data.loc[0, 'company_size'] = comp_size

# Set one-hot encoded features
employment_type_col = f'employment_type_{employment_type}'
if employment_type_col in input_data.columns:
    input_data.loc[0, employment_type_col] = 1

residence_col = f'employee_residence_{employee_residence}'
if residence_col in input_data.columns:
    input_data.loc[0, residence_col] = 1

location_col = f'company_location_{company_location}'
if location_col in input_data.columns:
    input_data.loc[0, location_col] = 1

# --- Predict button ---
if st.sidebar.button('🔍 Predict Salary'):
    try:
        prediction = model.predict(input_data)
        
        st.markdown(f"""
            <div class='prediction-box'>
                💰 Predicted Salary in USD: <br> ${prediction[0]:,.2f}
            </div>
        """, unsafe_allow_html=True)

        # Example visualization
        st.subheader("📊 Example Salary Trends by Company Size")
        sizes = ["Small", "Medium", "Large"]
        avg_salaries = [70000, 100000, 130000]

        fig, ax = plt.subplots()
        ax.bar(sizes, avg_salaries, color=["#66bb6a", "#43a047", "#2e7d32"])
        ax.set_ylabel("Average Salary (USD)")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Debug info:")
        st.write(f"Input shape: {input_data.shape}")
        st.write(f"Expected features: {len(expected_features)}")
        st.write("First few features:", expected_features[:5])
