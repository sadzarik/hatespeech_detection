import pandas as pd

# Шляхи до файлів
input_file = 'datasets/final_version_datasets_classes/class2_dataset.csv'
output_file = 'datasets/final_version_datasets_classes/class2_dataset.csv'

# Значення лейблу
target_label = 2

# 1. Завантажуємо файл
df = pd.read_csv(input_file, sep=';', encoding='utf-8-sig', on_bad_lines='skip')

# 2. Видаляємо порожні рядки у колонці 'text'
df.dropna(subset=['text'], inplace=True)
df = df[df['text'].str.strip() != '']

# 3. Додаємо лапки на початок і кінець тексту (якщо їх немає)
def ensure_quotes(text):
    text = text.strip()
    if not text.startswith('"'):
        text = '"' + text
    if not text.endswith('"'):
        text = text + '"'
    return text

df['text'] = df['text'].apply(ensure_quotes)

# 4. Додаємо колонку label або перезаписуємо її
df['label'] = target_label

# 5. Видаляємо колонку source (якщо є)
if 'source' in df.columns:
    df.drop(columns=['source'], inplace=True)

# 6. Зберігаємо
df.to_csv(output_file, sep=';', index=False, encoding='utf-8', quoting=1)

print(f"✅ Файл '{output_file}' збережено з {df.shape[0]} рядками та лейблом {target_label}.")
