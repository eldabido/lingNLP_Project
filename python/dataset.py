"""
Генерация датасета для обучения и оценки классификаторов никнеймов.

Источники:
  - Тест-кейсы из detector_test.go.
  - Записи из blacklist / whitelist в config.json.
  - Варианты с обфускацией (замена букв цифрами, разделители,
    смешение кириллицы и латиницы) - проверяем устойчивость к уловкам.
  - Расширенный список нейтральных слов, имён и никнеймов.
  - Дополнительные негативные примеры (оскорбления вне blacklist).

Размер: ~900–1000 записей.
"""

import json
import random
import csv
from pathlib import Path

# Тест-кейсы из Go-системы.
# 0 = запрещён, 1 = разрешён.
TEST_CASES: list[tuple[str, int]] = [
    # Здесь тесты.
]

# Дополнительные нейтральные никнеймы / слова (класс 1).
# Расширенный набор: имена, никнеймы, нейтральные слова, ложные срабатывания.
EXTRA_POSITIVE: list[str] = [
    # Убрал для цензуры.
]

# Дополнительные негативные примеры.
# Оскорбления и ругательства, не вошедшие в config.json blacklist.
EXTRA_NEGATIVE: list[str] = [

]

# Обфускация для негативных примеров.
OBFUSCATION_RULES: dict[str, list[str]] = {
    "а": ["4", "@", "a"],
    "е": ["3", "e"],
    "и": ["1", "i", "!"],
    "о": ["0", "o"],
    "у": ["y", "u"],
    "б": ["6", "b"],
    "т": ["7", "t", "+"],
    "с": ["$", "s", "c"],
    "х": ["x", "h"],
    "к": ["k"],
    "н": ["n"],
    "р": ["r"],
    "д": ["d"],
    "л": ["l"],
}

SEPARATORS = ["_", "-", ".", "*", " ", ""]

def obfuscate(word: str, seed: int = 0) -> str:
    """
    Применяет случайную обфускацию к слову:
    заменяет часть букв цифрами/символами, вставляет разделители.
    """
    rng = random.Random(seed)
    chars = list(word)

    # Заменяем 1–3 буквы на цифры/символы.
    for _ in range(rng.randint(1, 3)):
        idx = rng.randint(0, len(chars) - 1)
        ch = chars[idx]
        if ch in OBFUSCATION_RULES:
            chars[idx] = rng.choice(OBFUSCATION_RULES[ch])

    # Иногда вставляем разделитель между буквами.
    if rng.random() < 0.4 and len(chars) > 1:
        sep = rng.choice(SEPARATORS[:-1])
        pos = rng.randint(1, len(chars) - 1)
        chars.insert(pos, sep)

    # Иногда смешиваем регистр.
    if rng.random() < 0.3:
        chars = [c.upper() if rng.random() < 0.4 else c for c in chars]

    return "".join(chars)


def load_blacklist_words(config_path: Path) -> list[str]:
    """Возвращает слова из blacklist, помеченные как true."""
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return [word for word, active in config.get("blacklist", {}).items() if active]


def build_dataset(config_path: Path, obfuscations_per_word: int = 3) -> list[dict]:
    """
    Собирает итоговый датасет из всех источников (~900–1000 записей).
    """
    records: list[dict] = []
    seen: set[str] = set()

    def add(text: str, label: int, source: str) -> None:
        key = text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            records.append({"text": text.strip(), "label": label, "source": source})

    # Тест-кейсы.
    for text, label in TEST_CASES:
        add(text, label, "test_cases")

    # Нейтральные слова и ники.
    for word in EXTRA_POSITIVE:
        add(word, 1, "extra_positive")

    # Дополнительные негативные.
    for word in EXTRA_NEGATIVE:
        add(word, 0, "extra_negative")

    # Слова из blacklist config.json.
    blacklist_words = load_blacklist_words(config_path)
    for word in blacklist_words:
        add(word, 0, "blacklist")

    # Синтетические обфускации blacklist-слов.
    for i, word in enumerate(blacklist_words):
        for j in range(obfuscations_per_word):
            obf = obfuscate(word, seed=i * 100 + j)
            add(obf, 0, "obfuscated")

    # Синтетические обфускации дополнительных негативных.
    russian_extra = [w for w in EXTRA_NEGATIVE
                     if any('\u0400' <= c <= '\u04ff' for c in w) and '.' not in w]
    for i, word in enumerate(russian_extra):
        for j in range(2):  # 2 обфускации на слово
            obf = obfuscate(word, seed=5000 + i * 50 + j)
            add(obf, 0, "obfuscated")

    return records


def save_dataset(records: list[dict], output_path: Path) -> None:
    """Сохраняет датасет в CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "source"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Датасет сохранён: {output_path} ({len(records)} записей)")


def print_stats(records: list[dict]) -> None:
    """Статистика по датасету."""
    total = len(records)
    positive = sum(1 for r in records if r["label"] == 1)
    negative = total - positive
    by_source: dict[str, int] = {}
    for r in records:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1

    print(f"\n{'='*50}")
    print(f"Датасет: {total} записей")
    print(f"Разрешённые: {positive}  ({positive/total*100:.1f}%)")
    print(f"Запрещённые: {negative}  ({negative/total*100:.1f}%)")
    print(f"\nПо источнику:")
    for src, cnt in sorted(by_source.items()):
        print(f"{src:<20} {cnt}")

if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "go" / "config.json"
    output_path = Path(__file__).parent.parent / "data" / "dataset.csv"

    records = build_dataset(config_path)
    print_stats(records)
    save_dataset(records, output_path)
