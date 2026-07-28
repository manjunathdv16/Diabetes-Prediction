"""
Diabetes Prediction Model Module.

This module loads a pre-trained RandomForestClassifier model and provides
a prediction interface for diabetes risk assessment.
"""

import joblib
import json
import logging
import time
import datetime
import uuid
from config import (
    DIABETES_MODEL_FILE_PATH,
    DIABETES_MODEL_METADATA_FILE,
    MODEL_NAME,
    MODEL_VERSION,
)
from domino_data_capture.data_capture_client import DataCaptureClient

logger = logging.getLogger("diabetes_model")
logger.setLevel(logging.INFO)

# ============================== MODEL LOADING ==================================
# Load the pre-trained model from saved file
model = joblib.load(DIABETES_MODEL_FILE_PATH)

# Load metadata
with open(DIABETES_MODEL_METADATA_FILE, "r") as f:
    metadata = json.load(f)

# ==============================================================================
# Prediction Capture Configuration
# ==============================================================================

FEATURE_NAMES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
]

PREDICTION_NAMES = ["prediction"]

METADATA_NAMES = [
    "model_name",
    "model_version",
]

data_capture_client = DataCaptureClient(
    FEATURE_NAMES,
    PREDICTION_NAMES,
    METADATA_NAMES,
)

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

    request_id = str(uuid.uuid4())
    start = time.time()

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

    # ------------------------------------------------------------------
    # Capture prediction for Domino Model Monitoring
    # ------------------------------------------------------------------
    
    feature_values = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age,
    ]
    
    prediction_values = [
        int(prediction[0]),
    ]
    
    metadata_values = [
        metadata["model_name"],
        metadata["model_version"],
    ]
    
    event_id = str(uuid.uuid4())
    
    event_time = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    
    try:
        data_capture_client.capturePrediction(
            feature_values,
            prediction_values,
            metadata_values=metadata_values,
            event_id=event_id,
            timestamp=event_time,
            prediction_probability=[
                float(probability[0]),
                float(probability[1]),
            ],
            sample_weight=1.0,
        )
    
    except Exception as e:
        logger.exception(f"Prediction capture failed: {e}")

    latency_ms = round((time.time() - start) * 1000, 2)

    logging.info({
        "request_id": request_id,
        "prediction": int(prediction[0]),
        "latency_ms": latency_ms,
        "confidence": float(probability[1]),
        "model": "diabetes_model",
        "version": "1.0.0"
    })

    # Return result dictionary with prediction and confidence scores
    return {
        "request_id": request_id,
        "prediction": int(prediction[0]),
        "diabetic": bool(prediction[0]),
        "model": {
            "name": metadata["model_name"],
            "version": metadata["model_version"],
            "algorithm": metadata["algorithm"],
            "training_date": metadata["training_date"]
        },
        "confidence": {
            "non_diabetic": round(float(probability[0]), 4),
            "diabetic": round(float(probability[1]), 4)
        }
    }
