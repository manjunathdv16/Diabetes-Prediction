"""
DOMINO PLATFORM CONFIGURATION
Configuration for Diabetes Prediction API using Domino Model APIs.
All paths use environment variables — no hardcoded values.

These variables are set by the Domino platform when running jobs/apps
They specify where datasets, artifacts, and code are stored in the container

"""
import os

# Project name in Domino (used to construct dataset paths)
DOMINO_PROJECT_NAME = os.environ.get("DOMINO_PROJECT_NAME")

# Domino Datasets directory (where datasets are stored)
DOMINO_DATASETS_DIR = os.environ.get('DOMINO_DATASETS_DIR', '/mnt/data')
DATA_FILE_NAME = os.environ.get('DATA_FILE_NAME', 'diabetes.csv')
DATASETS_FILE_PATH = os.path.join(DOMINO_DATASETS_DIR, DOMINO_PROJECT_NAME, DATA_FILE_NAME)
MIGRATION_DATASET_NAME = os.environ.get('MIGRATION_DATASET_NAME', 'diabetes-data')

# Domino artifacts directory (where trained models and other artifacts are stored)
DOMINO_ARTIFACTS_DIR = os.environ.get('DOMINO_ARTIFACTS_DIR', '/mnt/artifacts')
MODELS_DIR = os.environ.get('MODELS_DIR', 'models')
MODEL_FILE_NAME = os.environ.get('MODEL_FILE_NAME', 'diabetes_model.pkl')
DIABETES_MODEL_FILE_PATH = os.path.join(DOMINO_ARTIFACTS_DIR, MODELS_DIR, MODEL_FILE_NAME)

# Domino Data Source name (for AWS S3 integration)
S3_DATASOURCE_NAME = os.environ.get('S3_DATASOURCE_NAME','diabetes-data')

# Legacy source (simulated)
LEGACY_SOURCE_DIR = os.environ.get('LEGACY_SOURCE_DIR', '/mnt/code/data')

# Snapshot schedule (cron-like)
SNAPSHOT_INTERVAL_HOURS = int(os.environ.get('SNAPSHOT_INTERVAL_HOURS', '24'))

# =============================================================================
# Model Training Configuration
# =============================================================================
# Training hyperparameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200

METADATA_FILE = os.environ.get("METADATA_FILE", "metadata.json")

DIABETES_MODEL_METADATA_FILE = os.path.join(
    DOMINO_ARTIFACTS_DIR,
    MODELS_DIR,
    METADATA_FILE
)

MODEL_NAME = os.environ.get("MODEL_NAME", "Diabetes Prediction Model")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")

TRAINING_SET_NAME = os.environ.get("TRAINING_SET_NAME", "diabetes-training-set-v3")
TRAINING_SET_DESCRIPTION = os.environ.get("TRAINING_SET_DESCRIPTION", "Training data for Diabetes Prediction Model")
TARGET_COLUMNS = ["Outcome"]

# =============================================================================
# API Server Configuration
# =============================================================================
APP_API_HOST = os.environ.get('APP_API_HOST', '0.0.0.0')
APP_API_PORT = int(os.environ.get('APP_API_PORT', 8888))
