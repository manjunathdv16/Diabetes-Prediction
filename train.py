import os
import io
import argparse
import joblib
import json
import sklearn
from datetime import datetime
import pandas as pd
from domino_data.training_sets.client import create_training_set_version
from domino_data.training_sets.model import MonitoringMeta

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
    TRAINING_SET_NAME,
    TRAINING_SET_DESCRIPTION,
    TARGET_COLUMNS,
)

# ============================= DATA LOADING ==============================

def load_dataset():
    """
    Load dataset from Domino Dataset / local filesystem.
    """
    # Read CSV file from configured path
    df = pd.read_csv(DATASETS_FILE_PATH)

    return df


def load_s3_dataset():
    """
    Load dataset from Domino Data Source (AWS S3).
    """
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

    return df


# ============================= MODEL TRAINING =============================

def train_model(df, data_source):
    """
    Train and save the diabetes prediction model.
    """
    try:
        
        monitoring_meta = MonitoringMeta(
            timestamp_columns=[],
            categorical_columns=["Outcome"],   # <-- important
            ordinal_columns=[
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ],
        )
        
        result = create_training_set_version(
            training_set_name=TRAINING_SET_NAME,
            df=df,
            description=TRAINING_SET_DESCRIPTION,
            target_columns=["Outcome"],
            monitoring_meta=monitoring_meta,
        )
        print(f"Registered TrainingSet '{TRAINING_SET_NAME}' version "
              f"{result.number} for Domino Model Monitoring.")

    except Exception as exc:
        print(f"WARNING: failed to register TrainingSet '{TRAINING_SET_NAME}' "
              f"for monitoring -- drift/quality metrics will not work until "
              f"this succeeds. Error: {exc}")
    
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

    # Create models directory
    os.makedirs(os.path.dirname(DIABETES_MODEL_FILE_PATH), exist_ok=True)

    # Save trained model
    joblib.dump(model, DIABETES_MODEL_FILE_PATH)

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
        "training_set": TRAINING_SET_NAME,
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

    # Load data from selected source
    if args.source == "aws":
        df = load_s3_dataset()
    else:
        df = load_dataset()

    # Train the model
    train_model(df, args.source)


if __name__ == "__main__":
    main()