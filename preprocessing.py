import re
import spacy

try:
    nlp = spacy.load("uk_core_news_sm")
except OSError:
    nlp = None
    print("Українську модель spaCy не знайдено. Для лематизації встановіть її через 'python -m spacy download uk_core_news_sm'.")

def clean_text(text: str) -> str:
    """
    Очищує текст від тегів користувачів, посилань, хештегів, зайвих пробілів.
    """
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text: str) -> str:
    """
    Лематизує текст, якщо завантажена модель spaCy.
    """
    if nlp is None:
        return text  # fallback: повертаємо текст як є
    doc = nlp(text)
    lemmatized = ' '.join([token.lemma_ for token in doc])
    return lemmatized
