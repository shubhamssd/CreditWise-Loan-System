
import joblib
import numpy as np

def load_model(model_path: str):
    return joblib.load(model_path)

def predict_loan(model, input_data: list):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return int(prediction[0])
