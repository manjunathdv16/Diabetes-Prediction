Absolutely. Since you're learning **Domino Data Lab and MLOps**, let's build a **real end-to-end project** that mirrors what you'd do in production.

## Project Structure

```text
diabetes-api/
│
├── train.py
├── model_diabetes.py
├── requirements.txt
├── data/
│   └── diabetes.csv
└── models/
    └── diabetes_model.pkl   (generated after training)
```

---

# Step 1: Download the dataset

Use the **Pima Indians Diabetes Dataset** from Kaggle.

Download:

[https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

After downloading, place:

```text
data/diabetes.csv
```

The CSV should have these columns:

```text
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Outcome
```

---

# Step 2: requirements.txt

```text
pandas
numpy
scikit-learn
joblib
```

Install:

```bash
pip install -r requirements.txt
```

---

# Step 3: train.py

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

```
Accuracy: 0.76
Model saved successfully.
```

---

# Step 4: model_diabetes.py

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

# Step 5: Create Domino Model API

Configure:

```
Script:
model_diabetes.py

Function:
predict
```

Deploy.

---

# Step 6: Test

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

Example response:

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

# Step 7: Verify the model

Before deploying, run:

```python
import joblib

model = joblib.load("models/diabetes_model.pkl")

print(model.n_features_in_)
```

Output:

```
8
```

This confirms the model expects the same eight features that your `predict()` function accepts.

---

## What you'll learn from this project

By completing this project, you'll practice the full ML lifecycle in Domino:

1. Load a real dataset.
2. Train a machine learning model.
3. Evaluate model performance.
4. Save the trained model with `joblib`.
5. Deploy it as a Domino Model API.
6. Invoke the API with `curl` or Postman.
7. Interpret predictions and confidence scores.

This is a solid foundation for understanding how Domino supports model development and deployment in a production-like workflow.
