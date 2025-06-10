from fastapi import FastAPI
from pydantic import BaseModel
from model import predict
from logger import log_request

app = FastAPI()

class TextInput(BaseModel):
    text: str


@app.post("/predict")
async def predict_aggression(input_data: TextInput):
    predicted_class, probabilities = predict(input_data.text)

    # Логування запиту
    log_request(input_data.text, predicted_class, probabilities)

    response = {
        "predicted_class": predicted_class,
        "probabilities": probabilities
    }
    return response

