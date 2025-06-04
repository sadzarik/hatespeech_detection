import pandas as pd

rows = []
bad_lines = []  # тут зберігатимемо номери й вміст тих рядків, які не вдалося правильно розпарсити

with open("annotation_dataset.csv", encoding="utf-8-sig") as f:
    header = f.readline()  # читаємо заголовок (очікуємо: clean_text,label)
    for lineno, line in enumerate(f, start=2):  # нумерація починається з 2, бо перший рядок — заголовок
        line = line.rstrip("\n")
        # Розділяємо саме за останньою комою
        parts = line.rsplit(",", 1)
        if len(parts) != 2:
            bad_lines.append((lineno, line))
            continue

        text_raw, label_raw = parts

        # Видаляємо зовнішні подвійні кавички, якщо вони є на початку та в кінці
        # і замінюємо всередині "" → "
        if text_raw.startswith('"') and text_raw.endswith('"'):
            text = text_raw[1:-1].replace('""', '"')
        else:
            text = text_raw

        # Перетворюємо лейбл на ціле число
        try:
            label = int(label_raw)
        except ValueError:
            bad_lines.append((lineno, line))
            continue

        rows.append((text, label))

# Якщо знайдено «погані» рядки, виведемо всі для діагностики
if bad_lines:
    print(f"Знайдено {len(bad_lines)} рядків, які не вдалося правильно розпарсити. Ось вони всі:\n")
    for ln, contents in bad_lines:
        print(f"Рядок {ln}: {contents}")
    print("\nПерегляньте ці рядки й виправте їх у CSV-файлі перед повторним запуском.\n")

# Створюємо DataFrame із тих рядків, що вдалося розпарсити
df = pd.DataFrame(rows, columns=["clean_text", "label"])

# Додаємо бінарний стовпець 'aggressive'
df["aggressive"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

# Порахуємо кількість агресивних/неагресивних
counts = df["aggressive"].value_counts().sort_index()
num_non_aggr = counts.get(0, 0)
num_aggr     = counts.get(1, 0)
total = len(df)

pct_non_aggr = num_non_aggr / total * 100
pct_aggr     = num_aggr / total * 100
ratio_aggr_to_non = num_aggr / num_non_aggr if num_non_aggr > 0 else float("inf")

print("=== РОЗПОДІЛ КОМЕНТАРІВ ===")
print(f"Агресивні (aggressive=1): {num_aggr} ({pct_aggr:.2f}%)")
print(f"Неагресивні (aggressive=0): {num_non_aggr} ({pct_non_aggr:.2f}%)")
print(f"Співвідношення (агресивні/неагресивні): {ratio_aggr_to_non:.2f}")

print("\n=== РОЗПОДІЛ ПО ПОЧАТКОВИМ LABEL ===")
print(df["label"].value_counts().sort_index().to_string())

# Додаткова статистика: середня довжина тексту
df["text_length"] = df["clean_text"].astype(str).apply(len)
avg_len_non = df[df["aggressive"] == 0]["text_length"].mean()
avg_len_aggr = df[df["aggressive"] == 1]["text_length"].mean()
print("\n=== ДОДАТКОВА ІНФОРМАЦІЯ ===")
print(f"Середня довжина неагресивного тексту: {avg_len_non:.2f} символів")
print(f"Середня довжина агресивного тексту: {avg_len_aggr:.2f} символів")

# --- ТУТ ЗБЕРІГАЄМО РОЗШИРЕНИЙ DataFrame У CSV ---
df.to_csv("annotation_dataset_with_aggression.csv", index=False, encoding="utf-8-sig")
