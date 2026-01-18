# src/predict.py
import numpy as np
import joblib
from src.logger import logger

def predict_loan(model, input_dict):
    try:
        encoders = joblib.load("models/encoders.pkl")

        # Encode categorical values
        for col, le in encoders.items():
            input_dict[col] = le.transform([input_dict[col]])[0]

        # ORDER MUST MATCH TRAINING DATAFRAME
        feature_order = [
            "Applicant_Income", "Coapplicant_Income", "Age",
            "Credit_Score", "Existing_Loans", "DTI_Ratio",
            "Savings", "Collateral_Value", "Loan_Amount", "Loan_Term",
            "Employment_Status", "Marital_Status", "Dependents",
            "Loan_Purpose", "Property_Area", "Education_Level",
            "Gender", "Employer_Category"
        ]

        input_array = np.array([input_dict[col] for col in feature_order]).reshape(1, -1)

        prediction = model.predict(input_array)
        logger.info("Prediction successful")
        return int(prediction[0])

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
