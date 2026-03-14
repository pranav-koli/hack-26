from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import os
from diet import router as diet_router

# =====================================
# CREATE APP
# =====================================

app = FastAPI(title="Disease Prediction API with Human Explainable AI")
app.include_router(diet_router)

# =====================================
# HEALTH CHECK (IMPORTANT FOR RAILWAY)
# =====================================

@app.get("/")
def home():
    return {
        "status": "Server Running",
        "api": "Disease Prediction API",
        "docs": "/docs"
    }

# =====================================
# CHECK FILES (DEBUG)
# =====================================

print("Files in deployment directory:", os.listdir())

# =====================================
# LOAD MODELS
# =====================================

try:

    heart_rf = joblib.load("heart_rf_model.pkl")
    heart_knn = joblib.load("heart_knn.pkl")
    heart_knn_scaler = joblib.load("heart_knn_scaler.pkl")
    HEART_COLUMNS = joblib.load("heart_columns.pkl")

    diabetes_lr = joblib.load("diabetes_lr_model.pkl")
    diabetes_knn = joblib.load("diabetes_knn_model.pkl")
    diabetes_scaler = joblib.load("diabetes_scaler.pkl")
    DIABETES_FEATURES = joblib.load("diabetes_features.pkl")

    heart_rf_explainer = shap.TreeExplainer(heart_rf)

    print("Models loaded successfully")

except Exception as e:
    print("Model loading failed:", e)
    raise e


# =====================================
# INPUT SCHEMAS
# =====================================

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


# =====================================
# HUMAN EXPLANATION FUNCTION
# =====================================

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


# =====================================
# HEART PREDICTION
# =====================================

@app.post("/predict/heart/{model_name}")
def predict_heart(model_name: str, data: HeartInput):

    try:

        X = pd.DataFrame([data.dict()])
        X = pd.get_dummies(X)
        X = X.reindex(columns=HEART_COLUMNS, fill_value=0)

        if model_name == "rf":
            prediction = heart_rf.predict(X)[0]

        elif model_name == "knn":
            X_scaled = heart_knn_scaler.transform(X)
            prediction = heart_knn.predict(X_scaled)[0]

        else:
            raise HTTPException(status_code=400, detail="Use rf or knn")

        return {
            "disease": "Heart Disease",
            "model_used": model_name,
            "prediction": int(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================
# HEART + EXPLAINABLE AI
# =====================================

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

        shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap_vals = shap_vals.reshape(-1)

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
                "method": "SHAP (Human-Friendly Explanation)",
                "reasons": explanations
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================
# DIABETES PREDICTION
# =====================================

@app.post("/predict/diabetes/{model_name}")
def predict_diabetes(model_name: str, data: DiabetesInput):

    try:

        X = pd.DataFrame([data.dict()])
        X = X[DIABETES_FEATURES]

        X_scaled = diabetes_scaler.transform(X)

        if model_name == "lr":
            prediction = diabetes_lr.predict(X_scaled)[0]

        elif model_name == "knn":
            prediction = diabetes_knn.predict(X_scaled)[0]

        else:
            raise HTTPException(status_code=400, detail="Use lr or knn")

        return {
            "disease": "Diabetes",
            "model_used": model_name,
            "prediction": int(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
