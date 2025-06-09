# Hate Speech Detection

Це проєкт для класифікації мовної агресії в українському тексті за допомогою BERT.

---

## 🚀 Як розгорнути проєкт

### 1️⃣ Клонування репозиторію

```
git clone https://github.com/sadzarik/hatespeech_detection.git
```
---
### 2️⃣ Створення віртуального середовища
```
python -m venv .venv
```
Активація:\
- Windows:
```
.venv\Scripts\activate
```
- Linux/Mac:
```
source .venv/bin/activate
```
---
### 3️⃣ Встановлення залежностей
**ВАЖЛИВО!** Torch із підтримкою CUDA 11.8 встановлювати окремо:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Далі:
```bash
pip install -r requirements.txt
```
---
### 4️⃣ Додатково встановити українську модель spaCy
```bash
python -m spacy download uk_core_news_sm
```
---
### 📝 Запуск
Для тренування моделі:
```bash
python train.py
```
Для запуску API:
```bash
uvicorn app:app --reload
```
Документація FastAPI буде доступна за [адресою](http://127.0.0.1:8000/docs).
---
### ⚠️ Примітка
Для моделей Transformers необхідний стабільний інтернет при першому завантаженні.

