import pickle
import numpy as np
import pandas as pd

def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = pickle.load(open('best_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except FileNotFoundError:
        print("Error: Model files not found. Please run heart_disease_model.py first.")
        return None, None

def predict_heart_disease():
    """Take user input and predict heart disease"""
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return
    
    print("\n=== Heart Disease Prediction System ===\n")
    
    # Get feature names from dataset
    df = pd.read_csv('dataset.csv')
    feature_names = df.drop('target', axis=1).columns.tolist()
    
    # Collect inputs
    patient_data = []
    print("Please enter the following information:")
    
    for feature in feature_names:
        while True:
            try:
                value = float(input(f"{feature}: "))
                patient_data.append(value)
                break
            except ValueError:
                print("Please enter a valid numeric value.")
    
    # Convert to numpy array and reshape
    patient_array = np.array(patient_data).reshape(1, -1)
    
    # Scale the input data
    patient_scaled = scaler.transform(patient_array)
    
    # Make prediction
    prediction = model.predict(patient_scaled)[0]
    prediction_proba = model.predict_proba(patient_scaled)[0]
    
    # Display result
    print("\n=== Prediction Results ===")
    if prediction == 1:
        print(f"Prediction: Heart Disease Detected")
        print(f"Confidence: {prediction_proba[1]:.2%}")
    else:
        print(f"Prediction: No Heart Disease Detected")
        print(f"Confidence: {prediction_proba[0]:.2%}")

if __name__ == "__main__":
    predict_heart_disease() 