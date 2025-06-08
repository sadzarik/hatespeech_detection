from transformers import BertTokenizer, BertForSequenceClassification
import torch
from preprocessing import clean_text, lemmatize_text

MODEL_PATH = "saved_model"
NUM_LABELS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Завантаження токенізатора та моделі
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=NUM_LABELS)
model.to(DEVICE)
model.eval()


def predict(text: str):
    """
    Виконує прогноз для одного тексту:
    1. Очистка
    2. Лематизація
    3. Токенізація
    4. Класифікація
    """
    # Очистка та лематизація
    text = clean_text(text)
    text = lemmatize_text(text)

    # Токенізація
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256  # Встановлюємо так само, як у тренуванні
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Прогноз
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
    return predicted_class, probs.squeeze().cpu().tolist()
