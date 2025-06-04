from fastapi import FastAPI
from pydantic import BaseModel
from preprocessing import clean_text, lemmatize_text
from model import predict

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_aggression(input_data: TextInput):
    text = clean_text(input_data.text)
    lemmatized = lemmatize_text(text)
    predicted_class, probabilities = predict(lemmatized)
    response = {
        "predicted_class": predicted_class,
        "probabilities": probabilities
    }
    return response
