from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI (change if your MLflow server is at a different address)
mlflow.set_tracking_uri("http://localhost:5000")

# MLflow model URI - update if needed
MODEL_URI = "models:/RandomForest/Production"

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info(f"Model loaded successfully from {MODEL_URI}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Initialize FastAPI app
app = FastAPI()

# Input data schema
class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {"message": "Welcome to the Churn Prediction API"}

@app.get("/health")
def health_check():
    logger.info("Health check endpoint accessed")
    status = "healthy" if model else "model not loaded"
    return {"status": status}

@app.post("/predict")
def predict(data: CustomerData):
    if not model:
        logger.error("Prediction requested but model is not loaded")
        return {"error": "Model not loaded"}

    logger.info(f"Prediction requested with input data: {data}")
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    result = int(prediction[0])
    logger.info(f"Prediction result: {result}")
    return {"churn_prediction": result}
