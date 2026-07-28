"""
Diabetes Prediction Model Module.

Loads the trained RandomForestClassifier model and exposes the
predict() function used by Domino Model API endpoints.

Also captures prediction events for Domino Model Monitoring.
"""

import os
import json
import time
import uuid
import datetime

import joblib
import pandas as pd

from config import (
    DIABETES_MODEL_FILE_PATH,
    DIABETES_MODEL_METADATA_FILE,
)

from domino_data_capture.data_capture_client import DataCaptureClient
from domino_data_capture import utils



# ==============================================================================
# Model Loading
# ==============================================================================

model = joblib.load(DIABETES_MODEL_FILE_PATH)

with open(DIABETES_MODEL_METADATA_FILE, "r") as f:
    metadata = json.load(f)


# ==============================================================================
# Prediction Capture Configuration
# ==============================================================================

FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

PREDICTION_NAMES = [
    "Outcome",  # must match train.py's TrainingSet target_columns exactly
]

METADATA_NAMES = [
    "model_name",
    "model_version",
]

data_capture_client = DataCaptureClient(
    FEATURE_NAMES,
    PREDICTION_NAMES,
    METADATA_NAMES,
)

# ==============================================================================
# Prediction Function
# ==============================================================================

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
    Predict diabetes risk.

    Returns:
        dict
    """

    request_id = str(uuid.uuid4())
    event_id = request_id

    start = time.time()

    # ------------------------------------------------------------------
    # Create DataFrame (prevents sklearn feature-name warning)
    # ------------------------------------------------------------------

    input_df = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree_function,
            age,
        ]],
        columns=FEATURE_NAMES,
    )

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0]

    prediction_value = int(prediction[0])

    latency_ms = round(
        (time.time() - start) * 1000,
        2,
    )

    # ------------------------------------------------------------------
    # Variables required by Domino Prediction Capture
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
        prediction_value,
    ]

    metadata_values = [
        metadata["model_name"],
        metadata["model_version"],
    ]

    event_time = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()

    # ------------------------------------------------------------------
    # Domino Prediction Capture
    # ------------------------------------------------------------------
    try:

        data_capture_client.capturePrediction(
            feature_values=feature_values,
            prediction_values=prediction_values,
            metadata_values=metadata_values,
            event_id=event_id,
            timestamp=event_time,
            prediction_probability=[
                float(probability[0]),
                float(probability[1]),
            ],
            sample_weight=[1.0],  # API expects Array[float], not a bare float
        )

    except Exception as exc:
        print(f"WARNING: capturePrediction failed for event_id={event_id} -- "
              f"this prediction will NOT appear in Model Monitor. Error: {exc}")

    # ------------------------------------------------------------------
    # API Response
    # ------------------------------------------------------------------

    response = {
        "request_id": request_id,
        "prediction": prediction_value,
        "diabetic": bool(prediction_value),
        "model": {
            "name": metadata["model_name"],
            "version": metadata["model_version"],
            "algorithm": metadata["algorithm"],
            "training_date": metadata["training_date"],
        },
        "confidence": {
            "non_diabetic": round(float(probability[0]), 4),
            "diabetic": round(float(probability[1]), 4),
        },
    }

    return response