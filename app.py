
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)

model = joblib.load("heart_disease_model.pkl")


class HeartData(BaseModel):
    Age: float
    Gender: float
    Blood_Pressure: float
    Cholesterol_Level: float
    Exercise_Habits: float
    Smoking: float
    Family_Heart_Disease: float
    Diabetes: float
    BMI: float
    High_Blood_Pressure: float
    Low_HDL_Cholesterol: float
    High_LDL_Cholesterol: float
    Alcohol_Consumption: float
    Stress_Level: float
    Sleep_Hours: float


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is Running!"}


@app.post("/predict")
def predict(data: HeartData):

    features = np.array([[
        data.Age,
        data.Gender,
        data.Blood_Pressure,
        data.Cholesterol_Level,
        data.Exercise_Habits,
        data.Smoking,
        data.Family_Heart_Disease,
        data.Diabetes,
        data.BMI,
        data.High_Blood_Pressure,
        data.Low_HDL_Cholesterol,
        data.High_LDL_Cholesterol,
        data.Alcohol_Consumption,
        data.Stress_Level,
        data.Sleep_Hours
    ]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease"

    return {
        "prediction": int(prediction),
        "result": result
    }
