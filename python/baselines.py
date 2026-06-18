"""
Готовые решения для обнаружения нецензурной лексики — базовые линии для сравнения.
Реализованные системы:
  1. DostoevskyBaseline     — FastText-модель для анализа тональности русского текста
                              (библиотека dostoevsky). Используем score класса
                              «negative» как proxy для токсичности.
  2. RuBERTToxicityBaseline — RuBERT-tiny, дообученный на классификацию токсичности
                              русских комментариев (2ch, pikabu, ok.ru).
                              HuggingFace: cointegrated/rubert-tiny-toxicity.
  3. DetoxifyBaseline       — multilingual BERT (XLM-R) от Unitary AI,
                              оценивает токсичность по шкале 0–1.
  4. SubstringBaseline      — простейший substring-поиск по blacklist (lower bound).

Все классы реализуют единый интерфейс: predict(texts) -> list[int].
0 = запрещён, 1 = разрешён.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import json

"""Базовый интерфейс для всех детекторов."""
class BaseDetector(ABC):
    name: str = "base"

    @abstractmethod
    def predict(self, texts: list[str]) -> list[int]:
        """Классифицирует никнеймы. 0 = запрещён, 1 = разрешён."""
        ...

    def predict_one(self, text: str) -> int:
        return self.predict([text])[0]


# RuBERT-tiny2-russian-sentiment
# HuggingFace: seara/rubert-tiny2-russian-sentiment
# RuBERT-tiny2, дообученный на sentiment-классификацию русских текстов.
# Классы: neutral (0), positive (1), negative (2).
# Мы используем класс «negative» как proxy для токсичности.
# Обучена на: Kaggle Russian News, Linis Crowd, RuReviews, RuSentiment.

class RuBERTSentimentBaseline(BaseDetector):
    """
    RuBERT-tiny2, fine-tuned на sentiment-классификацию русского текста.
    Класс negative -> скорее всего оскорбление.
    Модель работает через transformers.
    """

    name = "rubert-sentiment"
    MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"

    def __init__(self):
        self._available = False
        try:
            from transformers import pipeline
            print(f"rubert-tiny2-russian-sentiment: загрузка...")
            self._pipe = pipeline("text-classification", model=self.MODEL_NAME)
            self._available = True
            print("rubert-tiny2-russian-sentiment загружена")
        except Exception as e:
            print(f"[rubert-sentiment] Ошибка: {e}")

    def predict(self, texts: list[str]) -> list[int]:
        if not self._available:
            return [-1] * len(texts)
        try:
            preds = []
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                results = self._pipe(batch, truncation=True, max_length=128)
                for res in results:
                    # negative: запрещён (0), остальные: разрешён (1).
                    preds.append(0 if res["label"] == "negative" else 1)
            return preds
        except Exception as e:
            print(f"[rubert-sentiment] Ошибка predict: {e}")
            return [-1] * len(texts)


# RuBERT-tiny-toxicity
# HuggingFace: cointegrated/rubert-tiny-toxicity
# RuBERT-tiny, дообученный на токсичность русских комментариев.
# Классы: non-toxic, insult, obscenity, threat, dangerous.
# Мы считаем текст запрещённым, если non_toxic < 0.5.

class RuBERTToxicityBaseline(BaseDetector):
    """
    RuBERT-tiny, fine-tuned на классификацию токсичности русского текста.
    """

    name = "rubert-toxicity"
    MODEL_NAME = "cointegrated/rubert-tiny-toxicity"
    THRESHOLD = 0.5

    def __init__(self):
        self._available = False
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"rubert-tiny-toxicity: загрузка модели...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self._model.eval()
            # Получаем индекс класса non-toxic из config модели.
            self._labels = list(self._model.config.id2label.values())
            self._available = True
            print(f"rubert-tiny-toxicity загружена (классы: {self._labels})")
        except Exception as e:
            print(f"[rubert-toxicity] Ошибка: {e}")

    def predict(self, texts: list[str]) -> list[int]:
        if not self._available:
            return [-1] * len(texts)
        import torch
        try:
            preds = []
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=128, return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = torch.sigmoid(outputs.logits).cpu().numpy()

                for row in probs:
                    # row — массив вероятностей по классам.
                    # Ищем индекс non-toxic.
                    non_toxic_idx = None
                    for idx, label in enumerate(self._labels):
                        if "non" in label.lower() or "safe" in label.lower():
                            non_toxic_idx = idx
                            break
                    if non_toxic_idx is not None:
                        is_safe = row[non_toxic_idx] >= self.THRESHOLD
                    else:
                        # Если нет класса non-toxic, берём max из остальных.
                        is_safe = max(row) < self.THRESHOLD
                    preds.append(1 if is_safe else 0)
            return preds
        except Exception as e:
            print(f"[rubert-toxicity] Ошибка predict: {e}")
            return [-1] * len(texts)


# detoxify
# Multilingual BERT (XLM-RoBERTa), поддерживает русский.
# Порог toxicity ≥ 0.5.
class DetoxifyBaseline(BaseDetector):
    """
    Multilingual BERT-based модель от Unitary AI.
    Оценивает «токсичность» от 0.0 до 1.0.
    """

    name = "detoxify"
    THRESHOLD = 0.5

    def __init__(self, model_type: str = "multilingual"):
        self._available = False
        try:
            from detoxify import Detoxify
            print("detoxify: загрузка модели...")
            self.model = Detoxify(model_type)
            self._available = True
            print("detoxify загружена")
        except Exception as e:
            print(f"[detoxify] Ошибка: {e}")

    def predict(self, texts: list[str]) -> list[int]:
        if not self._available:
            return [-1] * len(texts)
        try:
            results = self.model.predict(texts)
            scores = results["toxicity"]
            return [0 if s >= self.THRESHOLD else 1 for s in scores]
        except Exception as e:
            print(f"[detoxify] Ошибка predict: {e}")
            return [-1] * len(texts)


# Substring baseline — нижняя граница.
class SubstringBaseline(BaseDetector):
    """
    Наивный детектор: проверяет вхождение слов из blacklist в lowercase текста.
    Без нормализации — нижняя граница (lower bound).
    """
    name = "substring"

    def __init__(self, config_path: Path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        self.blacklist: list[str] = [
            w for w, active in config.get("blacklist", {}).items() if active
        ]
        print(f"substring baseline ({len(self.blacklist)} слов)")

    def predict(self, texts: list[str]) -> list[int]:
        results = []
        for text in texts:
            lower = text.lower()
            found = any(bad in lower for bad in self.blacklist)
            results.append(0 if found else 1)
        return results


def get_baseline(name: str, config_path: Path | None = None) -> BaseDetector:
    """Создаёт детектор по имени."""
    if name == "rubert-sentiment":
        return RuBERTSentimentBaseline()
    elif name == "rubert-toxicity":
        return RuBERTToxicityBaseline()
    elif name == "detoxify":
        return DetoxifyBaseline()
    elif name == "substring":
        if config_path is None:
            raise ValueError("config_path обязателен для SubstringBaseline")
        return SubstringBaseline(config_path)
    else:
        raise ValueError(f"Неизвестный baseline: {name}")
