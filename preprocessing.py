import re
import spacy

try:
    nlp = spacy.load("uk_core_news_sm")
except OSError:
    nlp = None
    print("Українську модель spaCy не знайдено.")

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text: str) -> str:
    if nlp is None:
        return text  # fallback, якщо немає української моделі
    doc = nlp(text)
    lemmatized = ' '.join([token.lemma_ for token in doc])
    return lemmatized
