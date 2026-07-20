# Heart-disease-prediction-# Heart Disease Prediction API ❤️

A Machine Learning API built with FastAPI that predicts whether a person is at risk of heart disease.

## Features
- FastAPI
- Scikit-learn
- REST API
- JSON Response
- Render Deployment
- Swagger Documentation

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open:
http://127.0.0.1:8000/docs

## API Response

```json
{
  "prediction": 1,
  "result": "Heart Disease Detected"
}
```
