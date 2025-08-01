# 💼 Salary Prediction using Machine Learning

This project predicts employee salaries based on various job-related and company-specific features using machine learning models. It was developed using IBM Watson Studio and AutoAI.

## 📌 Project Overview

The goal of this project is to build a regression model that accurately predicts salaries in USD based on factors like:

- Work Year  
- Experience Level  
- Employment Type  
- Job Title  
- Employee Residence  
- Remote Work Ratio  
- Company Location  
- Company Size

The dataset used contains 245 rows and 11 columns.

---

## 🔍 Dataset Details

Each record in the dataset includes:

- work_year: Year of work (e.g., 2023)  
- experience_level: Junior / Mid / Senior / Executive  
- employment_type: Full-time, Part-time, Contract  
- job_title: Job designation  
- salary: Local currency salary  
- salary_currency: Currency code  
- salary_in_usd: Converted salary in USD (Target Variable)  
- employee_residence: Country of residence  
- remote_ratio: % of remote work  
- company_location: Location of employer  
- company_size: S / M / L

---

## ⚙ Tools & Technologies Used

- Python  
- IBM Watson Studio  
- AutoAI  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Lale  
- Matplotlib (for visualization)
- Streamlit

---

## 🧠 Machine Learning Approach

IBM AutoAI automatically performed the following:

- Data preprocessing  
- Feature engineering  
- Model selection (GradientBoostingRegressor , Random Forest)  
- Hyperparameter optimization  
- Evaluation

The best performing model was selected and used for predictions.

---

## 📝 Requirements
pip install -r requirements.txt
Streamlit run app.py
