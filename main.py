from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
from diet import router as diet_router

app = FastAPI(title="Disease Prediction API with Human Explainable AI")
app.include_router(diet_router)

# =========================
# LOAD MODELS
# =========================

heart_rf = joblib.load("models/heart_rf_model.pkl")
heart_knn = joblib.load("models/heart_knn.pkl")
heart_knn_scaler = joblib.load("models/heart_knn_scaler.pkl")
HEART_COLUMNS = joblib.load("models/heart_columns.pkl")

# Diabetes models
diabetes_lr = joblib.load("diabetes_lr_model.pkl")
diabetes_knn = joblib.load("diabetes_knn_model.pkl")
diabetes_rf = joblib.load("diabetes_rf_model.pkl")

diabetes_scaler = joblib.load("diabetes_scaler.pkl")
DIABETES_FEATURES = joblib.load("diabetes_features.pkl")

# SHAP Explainer
heart_rf_explainer = shap.TreeExplainer(heart_rf)

# =========================
# INPUT SCHEMAS
# =========================

class HeartInput(BaseModel):
    Age: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    MaxHR: int
    Oldpeak: float
    Sex: str
    ChestPainType: str
    RestingECG: str
    ExerciseAngina: str
    ST_Slope: str


class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


# =========================
# HUMAN EXPLANATION
# =========================

def human_explanation(feature, impact):

    direction = "increases" if impact > 0 else "reduces"

    feature_map = {
        "Cholesterol": "cholesterol level",
        "RestingBP": "blood pressure",
        "MaxHR": "heart rate",
        "Age": "age",
        "Oldpeak": "heart stress during exercise",
        "FastingBS_1": "blood sugar level",
        "ExerciseAngina_Y": "chest pain during exercise",
        "ST_Slope_Flat": "ECG heart pattern",
        "Sex_M": "male gender"
    }

    readable = feature_map.get(feature, feature.replace("_", " "))

    return f"Your {readable} {direction} the risk of heart disease."


# =========================
# HEART PREDICTION
# =========================

@app.post("/predict/heart/{model_name}")
def predict_heart(model_name: str, data: HeartInput):

    try:
        X = pd.DataFrame([data.dict()])
        X = pd.get_dummies(X)
        X = X.reindex(columns=HEART_COLUMNS, fill_value=0)

        if model_name == "rf":
            prediction = heart_rf.predict(X)[0]
            probability = heart_rf.predict_proba(X)[0][1]

        elif model_name == "knn":
            X_scaled = heart_knn_scaler.transform(X)
            prediction = heart_knn.predict(X_scaled)[0]
            probability = heart_knn.predict_proba(X_scaled)[0][1]

        else:
            raise HTTPException(status_code=400, detail="Use rf or knn")

        return {
            "disease": "Heart Disease",
            "model_used": model_name,
            "prediction": int(prediction),
            "probability": round(float(probability) * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# HEART + XAI
# =========================

@app.post("/predict-explain/heart/rf")
def predict_explain_heart_rf(data: HeartInput):

    try:
        X = pd.DataFrame([data.dict()])
        X = pd.get_dummies(X)
        X = X.reindex(columns=HEART_COLUMNS, fill_value=0)

        prediction = int(heart_rf.predict(X)[0])
        probability = float(heart_rf.predict_proba(X)[0][1])

        label = "Yes" if prediction == 1 else "No"

        shap_values = heart_rf_explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]

        explanations = []

        for feature, impact in zip(HEART_COLUMNS, shap_vals):
            explanations.append({
                "reason": human_explanation(feature, impact),
                "impact": round(float(impact), 4)
            })

        explanations = sorted(
            explanations,
            key=lambda x: abs(x["impact"]),
            reverse=True
        )[:5]

        return {
            "disease": "Heart Disease",
            "prediction": label,
            "probability": round(probability * 100, 2),
            "xai": {
                "method": "SHAP Human Explanation",
                "reasons": explanations
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# DIABETES PREDICTION
# =========================

@app.post("/predict/diabetes/{model_name}")
def predict_diabetes(model_name: str, data: DiabetesInput):

    try:

        X = pd.DataFrame([data.dict()])
        X = X.reindex(columns=DIABETES_FEATURES, fill_value=0)

        X_scaled = diabetes_scaler.transform(X)

        if model_name == "lr":
            prediction = diabetes_lr.predict(X_scaled)[0]
            probability = diabetes_lr.predict_proba(X_scaled)[0][1]

        elif model_name == "knn":
            prediction = diabetes_knn.predict(X_scaled)[0]
            probability = diabetes_knn.predict_proba(X_scaled)[0][1]

        elif model_name == "rf":
            prediction = diabetes_rf.predict(X_scaled)[0]
            probability = diabetes_rf.predict_proba(X_scaled)[0][1]

        else:
            raise HTTPException(status_code=400, detail="Use lr, knn or rf")

        return {
            "disease": "Diabetes",
            "model_used": model_name,
            "prediction": int(prediction),
            "probability": round(float(probability) * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
