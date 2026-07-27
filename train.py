import os
import io
import argparse
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import (
    DATA_FILE,
    DATA_SETS_FILE,
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
    DIABETES_MODEL_FILE,
    S3_DATASOURCE_NAME,
)
import config

# Display all configuration variables at startup for debugging
print("\n===== Configuration =====")

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
    df = pd.read_csv(DATA_SETS_FILE)

    print(f"Dataset loaded successfully from: {DATA_SETS_FILE}")
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
    object_store.download_fileobj(DATA_FILE, buffer)
    buffer.seek(0)

    # Load CSV from buffer
    df = pd.read_csv(buffer)

    # Get S3 bucket info for logging
    bucket = object_store.config["bucket"]
    subfolder = object_store.config["subfolder"]

    print(
        f"Dataset loaded successfully from AWS S3: "
        f"s3://{bucket}/{subfolder}/{DATA_FILE}"
    )
    print(f"Dataset Shape: {df.shape}")

    return df


# ============================= MODEL TRAINING =============================

def train_model(df):
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

    # Initialize Random Forest model with configured parameters
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )

    print("Training model...")

    # Train the model on training data
    model.fit(X_train, y_train)

    # Make predictions on test set
    predictions = model.predict(X_test)

    # Calculate model accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.3f}")

    # Create directory for model storage if it doesn't exist
    os.makedirs(os.path.dirname(DIABETES_MODEL_FILE), exist_ok=True)

    # Save trained model to disk
    joblib.dump(model, DIABETES_MODEL_FILE)

    print(f"Model saved successfully to: {DIABETES_MODEL_FILE}")


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
    train_model(df)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()