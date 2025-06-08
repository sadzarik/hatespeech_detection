import pandas as pd
import re

df_ukr = pd.read_csv("datasets/datasets_for_scrapping/ukr_tweets.csv", encoding='utf-8-sig', low_memory=False)

# === 1. Завантаження словника агресії ===
def load_aggressive_keywords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]

aggressive_keywords = load_aggressive_keywords('datasets/datasets_for_scrapping/aggressive_words.txt')

# === 2. Фільтрація, очищення ===
def is_valid_text(text):
    if pd.isna(text):
        return False
    text = str(text).strip().lower()

    # короткі або порожні
    if len(text) < 5:
        return False
    # тільки @теги
    if re.fullmatch(r'(@\w+\s*)+', text):
        return False
    # "для @..."
    if re.fullmatch(r'(для\s+)?(@\w+\s*)+', text):
        return False
    # містить посилання або картинку
    if 'http' in text or 'pic.twitter.com' in text or '#' in text:
        return False

    return True

def clean_text(text):
    text = str(text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === 3. Обробка корпусу ===
df_filtered = df_ukr[df_ukr['text'].apply(is_valid_text)].drop_duplicates(subset='text')
df_filtered['clean_text'] = df_filtered['text'].apply(clean_text)

# === 4. Призначення пустих лейблів для всіх рядків ===
df_filtered['label'] = ''  # Спочатку всім призначаємо пусті лейбли

# === 5. Класифікація агресивних твітів за словником ===
def classify_label(text):
    t = text.lower()
    return 3 if any(w in t for w in aggressive_keywords) else ''

# Зберігаємо агресивні твіти у окрему частину
df_aggr = df_filtered[df_filtered['clean_text'].apply(classify_label) == 3]

# === 6. Вибір 5000 нейтральних + 5000 агресивних ===
df_neut = df_filtered[df_filtered['label'] == ''][['clean_text', 'label']]

# Вибірка 5000 агресивних і 5000 нейтральних твітів
df_aggr_sample = df_aggr[['clean_text', 'label']].sample(n=5000, random_state=42, replace=False)
df_neut_sample = df_neut[['clean_text', 'label']].sample(n=5000, random_state=42, replace=False)

# === 7. Об’єднання: нейтральні + агресивні (агресія внизу) ===
df_final = pd.concat([df_neut_sample, df_aggr_sample], ignore_index=True)

# === 8. Збереження у CSV ===
df_final.to_csv("annotation_dataset.csv", index=False, encoding='utf-8-sig')
print("Готово: файл 'annotation_dataset.csv' містить 10 000 рядків")

print(f"Кількість українських твітів: {len(df_final)}")
print("Середня довжина твіту:", df_final['clean_text'].dropna().apply(len).mean())
print("Максимальна довжина твіту:", df_final['clean_text'].dropna().apply(len).max())
