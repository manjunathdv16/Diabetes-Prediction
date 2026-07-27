"""
Health endpoint for the Diabetes Prediction Model.
"""

import os
import json
import joblib
import platform
import sklearn

from config import (
    DIABETES_MODEL_FILE_PATH,
    DIABETES_MODEL_METADATA_FILE,
)

def health():
    """
    Health check endpoint.

    Returns:
        dict: Health status and model information.
    """

    status = "Healthy"
    issues = []

    # ------------------------------------------------------------------
    # Check model file
    # ------------------------------------------------------------------

    if not os.path.exists(DIABETES_MODEL_FILE_PATH):
        status = "Unhealthy"
        issues.append(f"Model file not found: {DIABETES_MODEL_FILE_PATH}")

    # ------------------------------------------------------------------
    # Check metadata file
    # ------------------------------------------------------------------

    if not os.path.exists(DIABETES_MODEL_METADATA_FILE):
        status = "Unhealthy"
        issues.append(f"Metadata file not found: {DIABETES_MODEL_METADATA_FILE}")

    metadata = {}

    if status == "Healthy":

        try:
            # Verify model can be loaded
            joblib.load(DIABETES_MODEL_FILE_PATH)

            # Verify metadata can be loaded
            with open(DIABETES_MODEL_METADATA_FILE) as f:
                metadata = json.load(f)

        except Exception as ex:
            status = "Unhealthy"
            issues.append(str(ex))

    return {
        "status": status,
        "model_loaded": status == "Healthy",
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "training_date": metadata.get("training_date"),
        "algorithm": metadata.get("algorithm"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "issues": issues,
    }