from transformers import BertTokenizer, BertForSequenceClassification
import torch

MODEL_PATH = "saved_model"
NUM_LABELS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Завантаження токенізатора та моделі
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=NUM_LABELS)
model.to(DEVICE)
model.eval()

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
    return predicted_class, probs.squeeze().cpu().tolist()
