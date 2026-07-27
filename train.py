import os
import io
import argparse
import joblib
import json
import sklearn
from datetime import datetime
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from config import (
    DATA_FILE_NAME,
    DATASETS_FILE_PATH,
    MODEL_NAME,
    MODEL_VERSION,
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
    DIABETES_MODEL_FILE_PATH,
    S3_DATASOURCE_NAME,
)

# Display all configuration variables at startup for debugging
print("\n===== Configuration =====")

import config
for name in dir(config):
    if name.isupper():
        print(f"{name:25} = {getattr(config, name)}")

print("=========================\n")

# ============================= DATA LOADING ==============================

def load_dataset():
    """
    Load dataset from Domino Dataset / local filesystem.
    """
    print("Loading dataset from Domino Dataset...")

    # Read CSV file from configured path
    df = pd.read_csv(DATASETS_FILE_PATH)

    print(f"Dataset loaded successfully from: {DATASETS_FILE_PATH}")
    print(f"Dataset Shape: {df.shape}")

    return df


def load_s3_dataset():
    """
    Load dataset from Domino Data Source (AWS S3).
    """
    print("Loading dataset from AWS S3 Data Source...")

    # Import Domino's S3 client
    from domino.data_sources import DataSourceClient
    import io
    import pandas as pd

    # Initialize client and get the data source
    object_store = DataSourceClient().get_datasource(S3_DATASOURCE_NAME)

    # Download file into memory buffer
    buffer = io.BytesIO()
    object_store.download_fileobj(DATA_FILE_NAME, buffer)
    buffer.seek(0)

    # Load CSV from buffer
    df = pd.read_csv(buffer)

    # Get S3 bucket info for logging
    bucket = object_store.config["bucket"]
    subfolder = object_store.config["subfolder"]

    print(
        f"Dataset loaded successfully from AWS S3: "
        f"s3://{bucket}/{subfolder}/{DATA_FILE_NAME}"
    )
    print(f"Dataset Shape: {df.shape}")

    return df


# ============================= MODEL TRAINING =============================

def train_model(df, data_source):
    """
    Train and save the diabetes prediction model.
    """

    # Separate features and target variable
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Initialize Random Forest model
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )

    print("Training model...")

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n========== Model Metrics ==========")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print("===================================\n")

    # Create models directory
    os.makedirs(os.path.dirname(DIABETES_MODEL_FILE_PATH), exist_ok=True)

    # Save trained model
    joblib.dump(model, DIABETES_MODEL_FILE_PATH)

    print(f"Model saved successfully to: {DIABETES_MODEL_FILE_PATH}")

    # ---------------------------------------------------------------------
    # Create metadata.json
    # ---------------------------------------------------------------------

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "algorithm": type(model).__name__,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_dataset": DATA_FILE_NAME,
        "training_source": data_source,
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_estimators": N_ESTIMATORS,
        "metrics": {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
        },
        "features": X.columns.tolist(),
        "artifact": {
            "schema_version": "1.0",
            "created_by": "train.py"
        },
    }

    metadata_file = os.path.join(
        os.path.dirname(DIABETES_MODEL_FILE_PATH),
        "metadata.json",
    )

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved successfully to: {metadata_file}")


# ============================== MAIN ENTRY ==================================

def main():
    """
    Main function to orchestrate the training pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Diabetes Prediction Model"
    )

    parser.add_argument(
        "source",
        nargs="?",
        default="dataset",
        choices=["dataset", "aws"],
        help="Training data source (default: dataset)",
    )

    args = parser.parse_args()

    print(f"\nUsing data source: {args.source}\n")

    # Load data from selected source
    if args.source == "aws":
        df = load_s3_dataset()
    else:
        df = load_dataset()

    # Train the model
    train_model(df, args.source)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()