import joblib

from config import DIABETES_MODEL_FILE

model = joblib.load(DIABETES_MODEL_FILE)


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
