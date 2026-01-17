
from src.data_preprocessing import load_data, handle_missing_values
from src.feature_engineering import encode_categorical_features, split_features_target
from src.model_training import split_data, train_model, save_model
from src.model_evaluation import evaluate_model

df = load_data("data/raw/loan_approval_data.csv")
df = handle_missing_values(df)

categorical_cols = [
    "Employment_Status", "Marital_Status", "Loan_Purpose",
    "Property_Area", "Education_Level", "Gender", "Employer_Category"
]

df = encode_categorical_features(df, categorical_cols)

X, y = split_features_target(df, "Loan_Approved")
X_train, X_test, y_train, y_test = split_data(X, y)

model = train_model(X_train, y_train)
accuracy, report = evaluate_model(model, X_test, y_test)

print("Accuracy:", accuracy)
print(report)

save_model(model, "models/loan_model.pkl")
