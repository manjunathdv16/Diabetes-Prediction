import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import DATA_FILE, TEST_SIZE, RANDOM_STATE, N_ESTIMATORS, MODELS_DIR, DIABETES_MODEL_FILE

# Load dataset
df = pd.read_csv(DATA_FILE)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

# Train model
model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.3f}")

# Save model
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(model, DIABETES_MODEL_FILE)

print("Model saved successfully.")
