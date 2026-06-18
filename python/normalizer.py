"""
Нормализатор текста. Используется для ML-моделей.
Этапы: замена цифр/символов, гомоглифов, транслитераций, мерджим повторы.
"""

import json
from pathlib import Path

# Минимальное допустимое число одинаковых символов подряд (аналог minLet из Go).
MIN_LET = 2

# Гомоглифы: латинские буквы, визуально похожие на кириллицу.
HOMOGLYPHS: dict[str, str] = {
    'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
    'x': 'х', 'y': 'у', 'k': 'к', 'b': 'б', 'm': 'м',
    'h': 'н', 'i': 'и', 'l': 'л',
}


class TextNormalizer:
    """
    Нормализует никнейм к единому виду для последующей проверки.
    Порядок этапов должен совпадать с Go-реализацией.
    """

    def __init__(self, config: dict):
        self.char_replacements: dict[str, str] = config.get("char_replacements", {})
        self.translit_map: dict[str, list[str]] = config.get("translit_map", {})
        self.long_patterns: dict[str, str] = config.get("long_patterns", {})

    def normalize(self, text: str) -> str:
        """Нормализует строку к кириллице без спецсимволов и повторов."""
        if not text:
            return ""
        text = text.lower()

        # Замена цифр и спецсимволов на буквы
        for src, dst in self.char_replacements.items():
            text = text.replace(src, dst)

        # Гомоглифы
        text = "".join(HOMOGLYPHS.get(ch, ch) for ch in text)
        # Транслитерация
        text = self._lat_to_cyr(text)
        # Оставляем только буквы
        text = "".join(ch for ch in text if ch.isalpha())
        # Убираем многократные повторы одной буквы
        return self._collapse_repeats(text)

    def _lat_to_cyr(self, text: str) -> str:
        """Заменяет многосимвольные паттерны (sh→ш), затем односимвольные (s→с)."""
        for lat, cyr in self.long_patterns.items():
            text = text.replace(lat, cyr)
        for cyr, lat_variants in self.translit_map.items():
            for lat in lat_variants:
                text = text.replace(lat, cyr)
        return text

    def _collapse_repeats(self, s: str) -> str:
        # Убираем повторы.
        result: list[str] = []
        prev = ""
        count = 0
        for ch in s:
            if ch == prev:
                count += 1
                if count < MIN_LET:
                    result.append(ch)
            else:
                prev = ch
                count = 1
                result.append(ch)
        return "".join(result)

def load_normalizer(config_path: str | Path) -> TextNormalizer:
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return TextNormalizer(config)
