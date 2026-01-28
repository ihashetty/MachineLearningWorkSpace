import joblib
import pandas as pd
from fastapi import FastAPI
 
app = FastAPI()
 
model = joblib.load("loan_pretrained.pkl")
 
@app.post("/predict")
def predict(data:dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"predicted_output":int(prediction[0])}