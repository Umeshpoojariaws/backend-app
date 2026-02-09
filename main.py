import mlflow
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from mlflow.tracking import MlflowClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The MLFLOW_TRACKING_URI will be injected by Kubernetes, so we remove the hardcoded value.

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    model_name = "weather-forecaster"
    
    try:
        # Get the latest model version from the "Production" stage
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not production_versions:
            logger.warning(f"No models found in 'Production' stage for '{model_name}'. Model will not be loaded.")
            model = None
            return
            
        latest_version = production_versions[0].version
        
        # Load the model after the server starts
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{latest_version}")
        logger.info(f"Successfully loaded model '{model_name}' version '{latest_version}' from 'Production' stage.")
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}'. Error: {e}")
        model = None

class PredictionPayload(BaseModel):
    today_temp: float
    humidity: float
    wind_speed: float

@app.get("/health")
def health_check(response: Response):
    if model is None:
        response.status_code = 503
        return {"status": "model_not_ready"}
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictionPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please promote a model to the 'Production' stage in MLflow.")
    data = pd.DataFrame([payload.dict()])
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}