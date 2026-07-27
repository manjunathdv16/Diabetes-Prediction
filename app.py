"""
Diabetes Prediction API - FastAPI Application.

This FastAPI application provides REST endpoints for diabetes risk prediction.
It integrates with the pre-trained RandomForest model from model_diabetes.py
and supports deployment on the Domino platform.

The API exposes the /predict endpoint for making predictions based on
patient health metrics.

Usage:
    python app.py              # Start the API server
    # Then visit: http://localhost:8888/docs (interactive API documentation)
    https://lscloud.product-team-sandbox.domino.tech/apps-internal/<app-id>/docs
    https://lscloud.product-team-sandbox.domino.tech/apps-internal/<app-id>/<endpoint> (Domino deployment)
"""

import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from config import APP_API_HOST, APP_API_PORT
from model_diabetes import predict as predict_diabetes

# =============================================================================
# DOMINO PLATFORM INTEGRATION
# =============================================================================
# Get the Domino host path from environment variable
# This is used as the root_path for the FastAPI app when deployed on Domino
path_prefix = os.environ["DOMINO_RUN_HOST_PATH"].rstrip("/")

print("DOMINO_RUN_HOST_PATH =", path_prefix)

# =============================================================================
# FASTAPI APPLICATION INITIALIZATION
# =============================================================================
# Create the FastAPI application
# root_path: Used by Domino to route requests to this app
# (e.g., requests to /domino-run-endpoint/predict are routed to /predict)
app = FastAPI(root_path=path_prefix)

print("FastAPI root_path =", app.root_path)
print("OpenAPI URL =", app.openapi_url)

# =============================================================================
# PYDANTIC MODELS (Input/Output Schemas)
# =============================================================================
# These models define the structure and validation of request/response data

class Item(BaseModel):
    """
    Generic item model (sample/demo).
    """
    name: str
    description: str
    price: float


class DiabetesPredictionInput(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float


@app.get("/")
async def root():
    """
    Root endpoint - Health check.

    Returns a simple message to confirm the API is running.
    """
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    """
    Sample endpoint - Get an item by ID.

    This is a demonstration endpoint.
    """
    return {"item_id": item_id}


@app.post("/items/")
async def create_item(item: Item):
    """
    Sample endpoint - Create a new item.

    This is a demonstration endpoint.
    """
    return item


@app.post("/predict")
async def predict(input_data: DiabetesPredictionInput):
    """
    Predict diabetes risk based on health metrics.
    
    Returns:
    - prediction: 0 or 1 (0 = non-diabetic, 1 = diabetic)
    - diabetic: boolean (True if diabetic, False otherwise)
    - confidence: probabilities for non_diabetic and diabetic classes
    """
    result = predict_diabetes(
        pregnancies=input_data.pregnancies,
        glucose=input_data.glucose,
        blood_pressure=input_data.blood_pressure,
        skin_thickness=input_data.skin_thickness,
        insulin=input_data.insulin,
        bmi=input_data.bmi,
        diabetes_pedigree_function=input_data.diabetes_pedigree_function,
        age=input_data.age
    )
    return result


@app.get("/env")
def env():
    """
    Debug endpoint - Get environment variables.

    Returns all environment variables containing "DOMINO" or "APP"
    for debugging and configuration verification.
    """
    return {
        k: v
        for k, v in os.environ.items()
        if "DOMINO" in k or "APP" in k
    }


@app.get("/debug")
def debug():
    """
    Debug endpoint - Get API configuration.

    Returns information about the FastAPI application setup
    including root_path and OpenAPI configuration.
    """
    return {
        "root_path": app.root_path,
        "openapi_url": app.openapi_url,
        "docs_url": app.docs_url,
        "run_host_path": os.environ.get("DOMINO_RUN_HOST_PATH")
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run the FastAPI server when script is executed directly.

    The API will be available at:
    - Main API: http://APP_API_HOST:APP_API_PORT/
    - Interactive Docs: http://APP_API_HOST:APP_API_PORT/docs
    - Alternative Docs: http://APP_API_HOST:APP_API_PORT/redoc
    """
    uvicorn.run(app, host=APP_API_HOST, port=APP_API_PORT)