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
# Domino Dataset, artifacts, AWS S3 bucket paths
DOMINO_DATASETS_DIR = os.environ.get('DOMINO_DATASETS_DIR', '/mnt/data')
DOMINO_ARTIFACTS_DIR = os.environ.get('DOMINO_ARTIFACTS_DIR', '/mnt/artifacts')
DATASET_NAME = os.environ.get('MIGRATION_DATASET_NAME', 'diabetes-data')
DATASET_PATH = os.path.join(DOMINO_DATASETS_DIR, DATASET_NAME)
S3_DATASOURCE_NAME = os.environ.get('S3_DATASOURCE_NAME','diabetes-data')

# Legacy source (simulated)
LEGACY_SOURCE_DIR = os.environ.get('LEGACY_SOURCE_DIR', '/mnt/code/data')

# Snapshot schedule (cron-like)
SNAPSHOT_INTERVAL_HOURS = int(os.environ.get('SNAPSHOT_INTERVAL_HOURS', '24'))

# =============================================================================
# Model Training Configuration
# =============================================================================
DATA_FILE = os.environ.get('DATA_FILE', 'diabetes.csv')
DATA_SETS_FILE = os.path.join(DOMINO_DATASETS_DIR, DOMINO_PROJECT_NAME, DATA_FILE)

MODELS_DIR = os.environ.get('MODELS_DIR', 'models')
MODEL_FILE = os.environ.get('MODEL_FILE', 'diabetes_model.pkl')
DIABETES_MODEL_FILE = os.path.join(DOMINO_ARTIFACTS_DIR, MODELS_DIR, MODEL_FILE)

# Training hyperparameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200

# =============================================================================
# API Server Configuration
# =============================================================================
API_HOST = "0.0.0.0"
API_PORT = 8888
