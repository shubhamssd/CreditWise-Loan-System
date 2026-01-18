import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
from src.predict import predict_loan

# Load model
model = joblib.load("models/loan_model.pkl")

st.set_page_config(page_title="CreditWise Loan Approval", layout="centered")

st.title("CreditWise Loan Approval System")
st.write("Enter applicant details to predict loan approval.")

Applicant_Income = st.number_input("Applicant Income", min_value=0)
Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0)
Age = st.number_input("Age", min_value=18)
Credit_Score = st.number_input("Credit Score", min_value=300, max_value=900)
Existing_Loans = st.number_input("Existing Loans", min_value=0)
DTI_Ratio = st.number_input("DTI Ratio", min_value=0.0)
Savings = st.number_input("Savings", min_value=0)
Collateral_Value = st.number_input("Collateral Value", min_value=0)
Loan_Amount = st.number_input("Loan Amount", min_value=0)
Loan_Term = st.number_input("Loan Term (Months)", min_value=6)
Employment_Status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Business"])
Marital_Status = st.selectbox("Marital Status", ["Married", "Single"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Loan_Purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Personal", "Business"])
Property_Area = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])
Education_Level = st.selectbox("Education Level", ["Graduate", "Postgraduate", "Undergraduate"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Employer_Category = st.selectbox("Employer Category", ["Govt", "Private", "Self"])


if st.button("Predict Loan Status"):
    input_data = {
    "Applicant_Income": Applicant_Income,
    "Coapplicant_Income": Coapplicant_Income,
    "Age": Age,
    "Credit_Score": Credit_Score,
    "Existing_Loans": Existing_Loans,
    "DTI_Ratio": DTI_Ratio,
    "Savings": Savings,
    "Collateral_Value": Collateral_Value,
    "Loan_Amount": Loan_Amount,
    "Loan_Term": Loan_Term,
    "Employment_Status": Employment_Status,
    "Marital_Status": Marital_Status,
    "Dependents": Dependents,
    "Loan_Purpose": Loan_Purpose,
    "Property_Area": Property_Area,
    "Education_Level": Education_Level,
    "Gender": Gender,
    "Employer_Category": Employer_Category
}


    result = predict_loan(model, input_data)

    if result == 1:
        st.success(" Loan Approved")
    else:
        st.error(" Loan Rejected")
