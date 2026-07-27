"""
Configuration for Diabetes Prediction API using Domino Model APIs.
All paths use environment variables — no hardcoded values.
"""
import os

# Domino Dataset paths
DOMINO_DATASETS_DIR = os.environ.get('DOMINO_DATASETS_DIR', '/mnt/data')
DATASET_NAME = os.environ.get('MIGRATION_DATASET_NAME', 'diabetes-data')
DATASET_PATH = os.path.join(DOMINO_DATASETS_DIR, DATASET_NAME)

# Legacy source (simulated)
LEGACY_SOURCE_DIR = os.environ.get('LEGACY_SOURCE_DIR', '/mnt/code/data')

# Snapshot schedule (cron-like)
SNAPSHOT_INTERVAL_HOURS = int(os.environ.get('SNAPSHOT_INTERVAL_HOURS', '24'))
