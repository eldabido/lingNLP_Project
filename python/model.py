"""
ML-модели для классификации никнеймов.

Подход: символьные n-граммы через TF-IDF — хорошо улавливают паттерны нецензурных слов даже при обфускации.
После нормализации модель работает с очищенным текстом.

Модели:
  - LogisticRegression - быстрая, интерпретируемая базовая линия.
  - LinearSVC - SVM, часто лучше на коротких текстах.
  - RandomForest - для сравнения ансамблевого метода.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.calibration import CalibratedClassifierCV

from normalizer import load_normalizer

# Пути к файлам.
CONFIG_PATH = Path(__file__).parent.parent / "go" / "config.json"
DATA_PATH = Path(__file__).parent.parent / "data" / "dataset.csv"
MODELS_DIR = Path(__file__).parent.parent / "results"

"""
Нормализация.
"""
def preprocess(texts: list[str], normalizer) -> list[str]:
    return [normalizer.normalize(t) for t in texts]


def build_pipeline(model_name: str) -> Pipeline:
    """
    Строит sklearn Pipeline: TF-IDF (char n-grams) + классификатор.

    Символьные биграммы-пятиграммы хорошо работают для обфусцированных слов,
    так как улавливают характерные последовательности букв.
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        # log(tf) вместо tf — сглаживает частоты.
        sublinear_tf=True,
        strip_accents="unicode",
    )

    if model_name == "logreg":
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "svm":
        # LinearSVC не имеет predict_proba, оборачиваем в CalibratedClassifierCV.
        clf = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
        )
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


class NickClassifier:
    """
    Обёртка над sklearn Pipeline с нормализацией.
    """

    def __init__(self, model_name: str, normalizer):
        self.model_name = model_name
        self.normalizer = normalizer
        self.pipeline = build_pipeline(model_name)

    def fit(self, texts: list[str], labels: list[int]) -> "NickClassifier":
        normalized = preprocess(texts, self.normalizer)
        self.pipeline.fit(normalized, labels)
        return self

    def predict(self, texts: list[str]) -> list[int]:
        normalized = preprocess(texts, self.normalizer)
        return self.pipeline.predict(normalized).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        normalized = preprocess(texts, self.normalizer)
        return self.pipeline.predict_proba(normalized)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "NickClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)


def cross_validate_model(
    model_name: str,
    texts: list[str],
    labels: list[int],
    normalizer,
    n_splits: int = 5,
) -> dict:
    """
    Оценка модели через k-fold кросс-валидацию.
    Возвращает словарь со средними метриками.
    """
    normalized = preprocess(texts, normalizer)
    pipeline = build_pipeline(model_name)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline,
        normalized,
        labels,
        cv=cv,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        return_train_score=False,
    )

    result = {
        "model": model_name,
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision_macro"].mean(),
        "recall": scores["test_recall_macro"].mean(),
        "f1": scores["test_f1_macro"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "f1_std": scores["test_f1_macro"].std(),
    }
    return result


def train_all_models(
    train_texts: list[str],
    train_labels: list[int],
    normalizer,
) -> dict[str, NickClassifier]:
    """Обучает все три модели и возвращает словарь."""
    models = {}
    for name in ["logreg", "svm", "rf"]:
        clf = NickClassifier(name, normalizer)
        clf.fit(train_texts, train_labels)
        models[name] = clf
        print(f"{name} обучена")
    return models


if __name__ == "__main__":
    # Быстрый тест: загружаем данные, запускаем кросс-валидацию.
    df = pd.read_csv(DATA_PATH)
    normalizer = load_normalizer(CONFIG_PATH)

    # Исключаем тест-кейсы из обучения.
    train_df = df[df["source"] != "test_cases"]
    texts = train_df["text"].tolist()
    labels = train_df["label"].tolist()

    print(f"\nОбучающая выборка: {len(texts)} примеров")
    print("Кросс-валидация (5-fold):\n")

    results = []
    for name in ["logreg", "svm", "rf"]:
        res = cross_validate_model(name, texts, labels, normalizer)
        results.append(res)
        print(
            f"  {name:<8}  F1={res['f1']:.3f} ± {res['f1_std']:.3f}"
            f"  Acc={res['accuracy']:.3f}"
            f"  P={res['precision']:.3f}  R={res['recall']:.3f}"
        )
