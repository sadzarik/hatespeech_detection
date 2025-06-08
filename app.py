from fastapi import FastAPI
from pydantic import BaseModel
from model import predict

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_aggression(input_data: TextInput):
    predicted_class, probabilities = predict(input_data.text)
    response = {
        "predicted_class": predicted_class,
        "probabilities": probabilities
    }
    return response
