import logging
from datetime import datetime
import os

# Створимо папку logs, якщо її ще немає
if not os.path.exists("logs"):
    os.makedirs("logs")

# Налаштуємо ім'я лог-файлу з таймштампом
log_file = f"logs/requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Конфігурація логера
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding='utf-8'
)

def log_request(text, predicted_class, probabilities):
    """
    Логує запит користувача та результат класифікації.
    """
    logging.info(f"Text: {text}")
    logging.info(f"Predicted Class: {predicted_class}")
    logging.info(f"Probabilities: {probabilities}")
    logging.info("-" * 50)
