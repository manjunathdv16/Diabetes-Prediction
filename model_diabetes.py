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
import logging
import datetime

import joblib

from config import (
    DIABETES_MODEL_FILE_PATH,
    DIABETES_MODEL_METADATA_FILE,
)

from domino_data_capture.data_capture_client import DataCaptureClient
from domino_data_capture import utils


# ==============================================================================
# Logging
# ==============================================================================

logger = logging.getLogger("diabetes_model")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)


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
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
]

PREDICTION_NAMES = [
    "prediction",
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

logger.info("=" * 70)
logger.info("Diabetes Prediction Model Loaded")
logger.info(f"Model Name       : {metadata['model_name']}")
logger.info(f"Model Version    : {metadata['model_version']}")
logger.info(f"Algorithm        : {metadata['algorithm']}")
logger.info(f"Training Date    : {metadata['training_date']}")
logger.info(f"Capture Dev Mode : {data_capture_client.is_dev_mode}")
logger.info(f"Scrape Location  : {utils.get_scrape_location()}")
logger.info("=" * 70)


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

    # ----------------------------------------------------------------------
    # Temporary debugging (keep until monitoring starts working)
    # ----------------------------------------------------------------------

    logger.info(
        f"PREDICTION_DATA_DIRECTORY={os.getenv('PREDICTION_DATA_DIRECTORY')}"
    )

    logger.info(
        f"HOSTNAME={os.getenv('HOSTNAME')}"
    )

    logger.info(
        f"SCRAPE_LOCATION={utils.get_scrape_location()}"
    )

    # ----------------------------------------------------------------------

    request_id = str(uuid.uuid4())
    event_id = request_id

    start = time.time()

    features = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age,
    ]]

    prediction = model.predict(features)
    probability = model.predict_proba(features)[0]

    prediction_value = int(prediction[0])

    latency_ms = round((time.time() - start) * 1000, 2)

    # ----------------------------------------------------------------------
    # Domino Prediction Capture
    # ----------------------------------------------------------------------

    try:

        capture_result = data_capture_client.capturePrediction(
            feature_values=feature_values,
            prediction_values=prediction_values,
            metadata_values=metadata_values,
            event_id=event_id,
            timestamp=event_time,
            prediction_probability=[
                float(probability[0]),
                float(probability[1]),
            ],
            sample_weight=1.0,
        )
    
        logger.info(f"Capture Result: {capture_result}")

        scrape_file = utils.get_scrape_location()
    
        logger.info(f"Scrape file path: {scrape_file}")
        logger.info(f"Scrape file exists: {os.path.exists(scrape_file)}")
        logger.info(
            f"Scrape dir exists: {os.path.exists('/var/scrape')}"
        )
        logger.info(
            f"Scrape dir writable: {os.access('/var/scrape', os.W_OK)}"
        )
        logger.info(
            f"Scrape file exists after capture: {os.path.exists(scrape_file)}"
        )
    
        logger.info(
            f"Prediction captured successfully | "
            f"request_id={request_id} "
            f"event_id={event_id} "
            f"prediction={prediction_value}"
        )
    
    except Exception:
        logger.exception("Prediction capture failed")

    # ----------------------------------------------------------------------
    # Request Logging
    # ----------------------------------------------------------------------

    logger.info(
        {
            "request_id": request_id,
            "prediction": prediction_value,
            "latency_ms": latency_ms,
            "confidence": float(probability[1]),
            "model": metadata["model_name"],
            "version": metadata["model_version"],
        }
    )

    # ----------------------------------------------------------------------
    # API Response
    # ----------------------------------------------------------------------

    return {
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
