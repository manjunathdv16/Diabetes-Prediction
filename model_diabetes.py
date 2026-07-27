"""
Diabetes Prediction Model Module.

This module loads a pre-trained RandomForestClassifier model and provides
a prediction interface for diabetes risk assessment.
"""

import joblib
from config import DIABETES_MODEL_FILE

# ============================== MODEL LOADING ==================================
# Load the pre-trained model from saved file
model = joblib.load(DIABETES_MODEL_FILE)


# ============================== PREDICTION FUNCTION ============================

def predict(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age,
):
    """
    Predict diabetes risk based on patient health metrics.

    Args:
        pregnancies: Number of times pregnant
        glucose: Plasma glucose concentration (mg/dL)
        blood_pressure: Diastolic blood pressure (mm Hg)
        skin_thickness: Triceps skin fold thickness (mm)
        insulin: 2-Hour serum insulin (mu U/ml)
        bmi: Body mass index (kg/m²)
        diabetes_pedigree_function: Genetic factor score
        age: Age in years

    Returns:
        dict: Prediction result with prediction class and confidence scores
    """

    # Make prediction on input features
    prediction = model.predict([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
    ]])

    # Get prediction probabilities
    probability = model.predict_proba([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
    ]])[0]

    # Return result dictionary with prediction and confidence scores
    return {
        "prediction": int(prediction[0]),
        "diabetic": bool(prediction[0]),
        "confidence": {
            "non_diabetic": round(float(probability[0]), 4),
            "diabetic": round(float(probability[1]), 4)
        }
    }
