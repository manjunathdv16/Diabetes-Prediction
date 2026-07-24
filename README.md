# Diabetes Prediction API using Domino Model API

This project demonstrates how to build, train, and deploy a Machine Learning model as a **Domino Model API**.

---

# Project Structure

```text
diabetes-api/
│
├── train.py
├── model_diabetes.py
├── requirements.txt
├── README.md
├── data/
│   └── diabetes.csv
└── models/
    └── diabetes_model.pkl
```

---

# Prerequisites

- Domino Data Lab
- Python 3.10+
- pip
- Git (optional)

---

# Step 1: Download the dataset

This project uses the **Pima Indians Diabetes Dataset**.

Download from:

https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

After downloading, place the dataset in:

```text
data/diabetes.csv
```

The dataset contains the following columns:

| Column |
|---------|
| Pregnancies |
| Glucose |
| BloodPressure |
| SkinThickness |
| Insulin |
| BMI |
| DiabetesPedigreeFunction |
| Age |
| Outcome |

---

# Step 2: Requirements.txt

```text
pandas
numpy
scikit-learn
joblib
```

---

# Step 3: Install Dependencies

Create a virtual environment (optional):

```bash
python -m venv venv
```

Activate it.

Install dependencies:

```bash
pip install -r requirements.txt
```

---


# Step 3: Train the Model

train.py

```python
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.3f}")

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/diabetes_model.pkl")

print("Model saved successfully.")
```
Run:

```bash
python train.py
```

Expected output:

```text
Accuracy: 0.76
Model saved successfully.
```

The script will automatically create:

```text
models/
    diabetes_model.pkl
```

---

# Verify the Saved Model

Run:

```python
import joblib

model = joblib.load("models/diabetes_model.pkl")

print(model.n_features_in_)
```

Expected output:

```text
8
```

This confirms the model expects 8 input features.

---

# model_diabetes.py

```python
import joblib

model = joblib.load("models/diabetes_model.pkl")


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

    prediction = model.predict([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
    ]])

    probability = model.predict_proba([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
    ]])[0]

    return {
        "prediction": int(prediction[0]),
        "diabetic": bool(prediction[0]),
        "confidence": {
            "non_diabetic": round(float(probability[0]), 4),
            "diabetic": round(float(probability[1]), 4)
        }
    }
```

---

# Create the Domino Model API

Create a new **Model API** in Domino using the following configuration:

| Setting | Value |
|----------|-------|
| Script | model_diabetes.py |
| Function | predict |

Deploy the model.

---

# Test the API

Example request:

```bash
curl \
'https://<domino-host>/models/<model-id>/latest/model' \
-u <API_KEY>:<API_KEY> \
-H "Content-Type: application/json" \
-d '{
  "data": {
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50
  }
}'
```

---

# Example Response

```json
{
  "result": {
    "prediction": 1,
    "diabetic": true,
    "confidence": {
      "non_diabetic": 0.14,
      "diabetic": 0.86
    }
  }
}
```

---

# Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| pregnancies | Integer | Number of pregnancies |
| glucose | Float | Plasma glucose concentration |
| blood_pressure | Float | Diastolic blood pressure |
| skin_thickness | Float | Triceps skin fold thickness |
| insulin | Float | 2-Hour serum insulin |
| bmi | Float | Body Mass Index |
| diabetes_pedigree_function | Float | Diabetes pedigree function |
| age | Integer | Age in years |

---

# Output

| Field | Description |
|-------|-------------|
| prediction | 0 = Non-Diabetic, 1 = Diabetic |
| diabetic | Boolean prediction |
| confidence | Prediction probabilities |

---

# Sync Changes to Domino Project

After training, remember to sync the generated model file back to your Domino project.

Generated file:

```text
models/diabetes_model.pkl
```

If using a Git-based project:

```bash
git add models/diabetes_model.pkl
git commit -m "Add trained diabetes model"
git push
```

If using a Files-based Domino project, use the Domino UI to sync or commit the workspace changes before deploying the Model API.

---

# Common Errors

## Model expects different number of features

Error:

```text
X has 8 features, but RandomForestClassifier is expecting 10 features.
```

Cause:

The deployed model was trained with a different dataset than the one expected by the API.

Solution:

- Retrain the model using the Pima Indians Diabetes dataset.
- Verify:

```python
print(model.n_features_in_)
```

Expected output:

```text
8
```

---

## File Not Found

Error:

```text
FileNotFoundError:
models/diabetes_model.pkl
```

Solution:

Run:

```bash
python train.py
```

Verify:

```bash
ls models
```

---

## Deployment Fails

Verify:

- Script name is correct.
- Function name is `predict`.
- `requirements.txt` includes all dependencies.
- `models/diabetes_model.pkl` exists in the project.

---

# Learning Outcomes

By completing this project, you will learn how to:

- Load a real-world dataset.
- Train a Random Forest classifier.
- Save a model using Joblib.
- Deploy a Python Function Endpoint in Domino.
- Test the endpoint using `curl` or Postman.
- Interpret prediction results and confidence scores.
- Troubleshoot common deployment issues.

---

# Next Steps

Consider extending the project with:

- Model Monitoring
- Model Versioning
- Batch Predictions
- Explainability (SHAP)
- MLflow Model Registry
- CI/CD pipeline for automated retraining and deployment
- Data drift detection
- Automated testing for the prediction API
