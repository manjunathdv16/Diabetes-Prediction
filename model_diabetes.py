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
import pandas as pd

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
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"
        )
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

    # ------------------------------------------------------------------
    # Temporary diagnostics
    # ------------------------------------------------------------------

    logger.info(
        f"PREDICTION_DATA_DIRECTORY={os.getenv('PREDICTION_DATA_DIRECTORY')}"
    )
    logger.info(
        f"HOSTNAME={os.getenv('HOSTNAME')}"
    )
    logger.info(
        f"SCRAPE_LOCATION={utils.get_scrape_location()}"
    )

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

        logger.info("=" * 70)
        logger.info("Prediction Capture Executed")
        logger.info("=" * 70)

        logger.info(f"Capture Result : {capture_result}")

        scrape_file = utils.get_scrape_location()

        logger.info(f"Scrape File : {scrape_file}")

        logger.info(
            f"Scrape Directory Exists : "
            f"{os.path.exists('/var/scrape')}"
        )

        logger.info(
            f"Scrape Directory Writable : "
            f"{os.access('/var/scrape', os.W_OK)}"
        )

        logger.info(
            f"Scrape File Exists : "
            f"{os.path.exists(scrape_file)}"
        )

        if os.path.exists(scrape_file):

            logger.info(
                f"Scrape File Size : "
                f"{os.path.getsize(scrape_file)} bytes"
            )

            try:

                with open(scrape_file, "r") as f:

                    lines = f.readlines()

                logger.info(
                    f"Scrape File Lines : {len(lines)}"
                )

                if lines:
                    logger.info(
                        "Last Scrape Record:"
                    )
                    logger.info(lines[-1].strip())

            except Exception:
                logger.exception(
                    "Unable to read scrape file."
                )

        else:

            logger.warning(
                "Prediction capture completed but "
                "scrape file was not created."
            )

        logger.info(
            f"Request ID : {request_id}"
        )

        logger.info(
            f"Event ID : {event_id}"
        )

        logger.info(
            f"Prediction : {prediction_value}"
        )

        logger.info(
            f"Probability : "
            f"{float(probability[1]):.4f}"
        )

        logger.info("=" * 70)

    except Exception:

        logger.exception(
            "Prediction capture failed."
        )

    # ------------------------------------------------------------------
    # Request Logging
    # ------------------------------------------------------------------

    logger.info("=" * 70)
    logger.info("Prediction Request Completed")
    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "event_id": event_id,
                "prediction": prediction_value,
                "diabetic": bool(prediction_value),
                "confidence": {
                    "non_diabetic": round(float(probability[0]), 4),
                    "diabetic": round(float(probability[1]), 4),
                },
                "latency_ms": latency_ms,
                "model": metadata["model_name"],
                "version": metadata["model_version"],
            },
            indent=2,
        )
    )
    logger.info("=" * 70)

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