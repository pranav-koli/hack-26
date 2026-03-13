from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# =====================
# REQUEST MODEL
# =====================
class DietRequest(BaseModel):
    age: int
    gender: str
    height: float
    weight: float
    activity_level: str
    predicted_disease: str


# =====================
# HELPER FUNCTIONS
# =====================
def calculate_bmi(height, weight):
    h_m = height / 100
    return round(weight / (h_m * h_m), 2)


def calorie_need(activity, gender):
    base = 2500 if gender.upper() == "M" else 2000

    if activity == "low":
        return base - 300
    elif activity == "high":
        return base + 300
    return base


def generate_diet(disease, bmi):
    disease = disease.lower()

    if disease == "heart":
        return [
            "Reduce oily and fried foods",
            "Eat fruits like apple, papaya",
            "Choose whole grains over refined grains",
            "Limit salt intake",
            "Drink plenty of water"
        ]

    if disease == "diabetes":
        return [
            "Avoid sugar and sweets",
            "Eat high-fiber foods",
            "Prefer complex carbohydrates",
            "Eat smaller meals frequently"
        ]

    if bmi > 25:
        return [
            "High protein diet",
            "Avoid junk food",
            "Increase physical activity"
        ]

    return [
        "Balanced diet",
        "Home cooked meals",
        "Seasonal fruits"
    ]


# =====================
# API ENDPOINT
# =====================
@router.post("/diet/from-prediction")
def personalized_diet(data: DietRequest):

    bmi = calculate_bmi(data.height, data.weight)
    calories = calorie_need(data.activity_level.lower(), data.gender)
    diet = generate_diet(data.predicted_disease, bmi)

    return {
        "BMI": bmi,
        "Daily_Calories": calories,
        "Goal": "Disease Management & Healthy Lifestyle",
        "Diet": diet
    }