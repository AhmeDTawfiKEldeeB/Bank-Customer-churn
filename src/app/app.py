from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

from src.serving.inference import predict

app = FastAPI(
    title="Bank Customer Churn API",
    description="FastAPI + Gradio churn prediction",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok"}

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

@app.post("/predict")
def predict_api(data: CustomerData):
    try:
        return predict(data.dict())
    except Exception as e:
        return {"error": str(e)}

# === Gradio UI ===
def gradio_interface(
    CreditScore, Geography, Gender, Age, Tenure,
    Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
):
    payload = {
        "CreditScore": int(CreditScore),
        "Geography": Geography,
        "Gender": Gender,
        "Age": int(Age),
        "Tenure": int(Tenure),
        "Balance": float(Balance),
        "NumOfProducts": int(NumOfProducts),
        "HasCrCard": int(HasCrCard),
        "IsActiveMember": int(IsActiveMember),
        "EstimatedSalary": float(EstimatedSalary),
    }
    return str(predict(payload))

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Number(label="Credit Score"),
        gr.Dropdown(["France","Spain","Germany"], label="Geography"),
        gr.Dropdown(["Male","Female"], label="Gender"),
        gr.Number(label="Age"),
        gr.Number(label="Tenure (months)"),
        gr.Number(label="Balance"),
        gr.Number(label="NumOfProducts"),
        gr.Number(label="HasCrCard (0/1)"),
        gr.Number(label="IsActiveMember (0/1)"),
        gr.Number(label="EstimatedSalary"),
    ],
    outputs="text",
    title="Bank Churn Predictor"
)

app = gr.mount_gradio_app(app, demo, path="/ui")
