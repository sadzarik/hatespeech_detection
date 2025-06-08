# ds_inspect.py
import itertools

with open("datasets/datasets_for_scrapping/annotation_dataset.csv", encoding="utf-8-sig") as f:
    for i, line in enumerate(f, start=1):
        if i == 2065:
            print("=== Рядок 2065 ===")
            print(line)
            break
