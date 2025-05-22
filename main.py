from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow.pyfunc
import joblib
import time
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response



app = FastAPI(title="Bank Customer Churn Prediction API")

#model_name = "bank_churn_model"  
#model_stage = "Production"

#model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "request_count", "Total number of requests", ["method", "endpoint"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency in seconds", ["endpoint"]
)

# Define the input schema
model = joblib.load("model.pkl")

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

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(process_time)

    return response

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: CustomerData):
    input_df = data.dict()
    input_df = [input_df]  
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}

# Prometheus metrics endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")