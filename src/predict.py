import joblib
import pandas as pd

MODEL_PATH = "models/model.joblib"

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def create_customer():
    customer = {
        "SeniorCitizen": [0],
        "tenure": [12],
        "MonthlyCharges": [70.0],
        "TotalCharges": [840.0],
        "gender": ["Female"],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["Yes"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
    }

    customer_df = pd.DataFrame(customer)
    return customer_df

def make_prediction(model, customer):
    prediction = model.predict(customer)
    probabilities = model.predict_proba(customer)

    predicted_class = prediction[0]
    churn_probability = probabilities[0][1] 

    return predicted_class, churn_probability

def print_result(customer_df, predicted_class, churn_probability):
    print("Customer Data: ")
    print(customer_df)

    print(f"Predicted class: {predicted_class}")
    
    print(f"\nChurn probability: {churn_probability:.2%}")
    print()

    if predicted_class == 1:
        print("Result: Customer is likely to churn")
    else:
        print("Result: Customer is not likely to churn")


def main():
    model = load_model(MODEL_PATH)
    customer = create_customer()
    predicted_class, churn_probability = make_prediction(model, customer)
    print_result(customer, predicted_class, churn_probability)

if __name__ == "__main__":
    main()