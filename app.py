import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Initialize the FastAPI application
app = FastAPI()

# Load the MLflow model
model_name = "Best_Model"
model_version = "1"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

# Define the input data structure
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add more features as needed based on your model

# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Make predictions using the loaded model
    predictions = model.predict(input_data)

    # Return predictions as JSON
    return {"predictions": predictions.tolist()}

# Optional: Define a root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API!"}