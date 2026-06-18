"""
Сравнительный анализ систем обнаружения нецензурных никнеймов.

Что сравниваем:
  - Go rule-based система (метрики из go_metrics.csv, предсказания из go_predictions.csv)
  - Наши ML-модели: LogReg, SVM, RandomForest (char n-gram TF-IDF)
  - Готовые решения: rubert-sentiment, rubert-tiny-toxicity, detoxify, substring baseline
"""

import csv
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from normalizer import load_normalizer
from dataset import build_dataset, TEST_CASES
from model import NickClassifier, train_all_models, cross_validate_model
from baselines import get_baseline

warnings.filterwarnings("ignore")

# Пути
ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "go" / "config.json"
DATA_PATH = ROOT / "data" / "dataset.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Названия для таблицы.
MODEL_LABELS = {
    "go_rule_based":    "Go (rule-based)",
    "logreg":           "LogReg (наша)",
    "svm":              "SVM (наша)",
    "rf":               "RandomForest (наша)",
    "rubert-sentiment":  "rubert-sentiment",
    "rubert-toxicity":  "rubert-tiny-toxicity",
    "detoxify":         "detoxify (BERT)",
    "substring":        "substring (baseline)",
}

# Обфусцированные тест-кейсы (цифры, спецсимволы, смешение алфавитов).
OBFUSCATED_NICKS = {

}

# Результаты Go rule-based системы (два файла от TestDetectorCSV)

GO_METRICS_PATH = ROOT / "results" / "go_metrics.csv"
GO_PREDICTIONS_PATH = ROOT / "results" / "go_predictions.csv"


def _load_go_metrics() -> dict | None:
    """
    Загружает итоговые метрики Go из go_metrics.csv.
    Возвращает {"all": {accuracy, precision, recall, f1}, "obfuscated": {...}}.
    """
    if not GO_METRICS_PATH.exists():
        return None
    result = {}
    with open(GO_METRICS_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subset = row["subset"].strip()
            result[subset] = {
                "accuracy":  float(row["accuracy"]),
                "precision": float(row["precision"]),
                "recall":    float(row["recall"]),
                "f1":        float(row["f1"]),
            }
    return result


def _load_go_predictions(texts: list[str]) -> list[int] | None:
    """
    Загружает пословные предсказания Go из go_predictions.csv.
    Возвращает список результатов (0/1) для каждого текста из texts.
    """
    if not GO_PREDICTIONS_PATH.exists():
        return None
    go_map: dict[str, int] = {}
    with open(GO_PREDICTIONS_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nick = row["nick"]
            is_allowed = row["got"].strip().lower() == "true"
            go_map[nick] = 1 if is_allowed else 0
    return [go_map.get(t, 1) for t in texts]


# Вспомогательные функции.

def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Считает accuracy, precision, recall, F1 (macro). Пропускает -1."""
    valid = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yp != -1]
    if not valid:
        return {"accuracy": None, "precision": None, "recall": None, "f1": None, "n": 0}
    yt, yp = zip(*valid)
    return {
        "accuracy":  accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, average="macro", zero_division=0),
        "recall":    recall_score(yt, yp, average="macro", zero_division=0),
        "f1":        f1_score(yt, yp, average="macro", zero_division=0),
        "n":         len(yt),
    }


def fmt(val) -> str:
    """Форматирует число для таблицы. None и NaN → 'N/A'."""
    if val is None:
        return "N/A"
    try:
        if math.isnan(val):
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"
    return f"{val:.3f}"


def split_test_cases():
    """Разделяет тест-кейсы на все / обфусцированные."""
    all_texts = [t for t, _ in TEST_CASES]
    all_labels = [l for _, l in TEST_CASES]

    obf_texts, obf_labels = [], []
    for text, label in TEST_CASES:
        if text in OBFUSCATED_NICKS:
            obf_texts.append(text)
            obf_labels.append(label)

    return all_texts, all_labels, obf_texts, obf_labels


# Главная функция.
def main():
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ ДЕТЕКТОРОВ НИКНЕЙМОВ")
    print("=" * 60)

    # Данные.
    print("\n[1/4] Подготовка данных...")
    records = build_dataset(CONFIG_PATH)
    df = pd.DataFrame(records)

    train_df = df[df["source"] != "test_cases"].copy()
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    pos = sum(train_labels)
    neg = len(train_labels) - pos
    print(f"Обучение: {len(train_texts)} примеров ({pos} разрешённых, {neg} запрещённых)")

    all_texts, all_labels, obf_texts, obf_labels = split_test_cases()
    print(f"Тест: {len(all_texts)} кейсов, из них обфусцированных: {len(obf_texts)}")

    # ML-модели.
    normalizer = load_normalizer(CONFIG_PATH)
    print("\n[2/4] Обучение ML-моделей...")
    ml_models = train_all_models(train_texts, train_labels, normalizer)

    # Baselines.
    print("\n[3/4] Загрузка готовых решений...")
    baselines = {
        "substring":        get_baseline("substring", CONFIG_PATH),
        "rubert-sentiment":  get_baseline("rubert-sentiment"),
        "rubert-toxicity":  get_baseline("rubert-toxicity"),
        "detoxify":         get_baseline("detoxify"),
    }

    # Оценка.
    print("\n[4/4] Оценка на тест-кейсах...\n")

    # Go-результаты.
    go_metrics = _load_go_metrics()
    go_preds_all = _load_go_predictions(all_texts)
    go_preds_obf = _load_go_predictions(obf_texts)

    if go_metrics:
        print(f"Go-метрики: {GO_METRICS_PATH}")
    if go_preds_all:
        print(f"Go-предсказания: {GO_PREDICTIONS_PATH}")

    # Собираем предсказания ML и baseline систем.
    predictions: dict[str, list[int]] = {}

    if go_preds_all:
        predictions["go_rule_based"] = go_preds_all

    for name, clf in ml_models.items():
        predictions[name] = clf.predict(all_texts)

    for name, bl in baselines.items():
        predictions[name] = bl.predict(all_texts)

    # Предсказания для обфусцированных.
    predictions_obf: dict[str, list[int]] = {}
    if go_preds_obf:
        predictions_obf["go_rule_based"] = go_preds_obf
    for name, clf in ml_models.items():
        predictions_obf[name] = clf.predict(obf_texts) if obf_texts else []
    for name, bl in baselines.items():
        predictions_obf[name] = bl.predict(obf_texts) if obf_texts else []

    # Таблица результатов.
    all_results = []

    # Go-метрики из go_metrics.csv.
    if go_metrics:
        go_all = go_metrics.get("all", {})
        go_obf = go_metrics.get("obfuscated", {})
        all_results.append({
            "system":  MODEL_LABELS["go_rule_based"],
            "f1_all":  go_all.get("f1"),
            "acc_all": go_all.get("accuracy"),
            "p_all":   go_all.get("precision"),
            "r_all":   go_all.get("recall"),
            "f1_obf":  go_obf.get("f1"),
            "acc_obf": go_obf.get("accuracy"),
        })
    else:
        all_results.append({
            "system": MODEL_LABELS["go_rule_based"],
            "f1_all": None, "acc_all": None, "p_all": None,
            "r_all": None, "f1_obf": None, "acc_obf": None,
        })

    # ML-модели и baselines — метрики считаем из предсказаний.
    system_order = ["logreg", "svm", "rf",
                    "rubert-sentiment", "rubert-toxicity", "detoxify", "substring"]

    for sys_name in system_order:
        if sys_name not in predictions:
            continue
        m_all = compute_metrics(all_labels, predictions[sys_name])
        m_obf = compute_metrics(obf_labels, predictions_obf.get(sys_name, []))
        all_results.append({
            "system":  MODEL_LABELS.get(sys_name, sys_name),
            "f1_all":  m_all["f1"],
            "acc_all": m_all["accuracy"],
            "p_all":   m_all["precision"],
            "r_all":   m_all["recall"],
            "f1_obf":  m_obf["f1"],
            "acc_obf": m_obf["accuracy"],
        })

    results_df = pd.DataFrame(all_results)

    print("=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"\n  {'Система':<23} {'F1 (все)':<11} {'Acc (все)':<11} "
          f"{'Prec':<8} {'Recall':<8} {'F1 (обф.)':<11} {'Acc (обф.)'}")
    print("  " + "-" * 80)
    for _, row in results_df.iterrows():
        print(
            f"  {row['system']:<23} {fmt(row['f1_all']):<11} {fmt(row['acc_all']):<11}"
            f" {fmt(row['p_all']):<8} {fmt(row['r_all']):<8}"
            f" {fmt(row['f1_obf']):<11} {fmt(row['acc_obf'])}"
        )
    print("  " + "=" * 80)

    # Кросс-валидация ML
    print("\nКросс-валидация ML-моделей (5-fold, обучающая выборка):")
    print(f"{'Модель':<10} {'F1':<14} {'Accuracy':<11} {'Precision':<11} Recall")
    print("  " + "-" * 60)
    for name in ["logreg", "svm", "rf"]:
        res = cross_validate_model(name, train_texts, train_labels, normalizer)
        print(
            f"{name:<10} {res['f1']:.3f}±{res['f1_std']:.3f}   "
            f"{res['accuracy']:.3f}      {res['precision']:.3f}      {res['recall']:.3f}"
        )

    # Сложные случаи, где системы расходятся.
    print("\nДетальный разбор:")
    cols = ["logreg", "svm", "substring"]
    col_labels = ["LogReg", "SVM", "Substr"]
    if "go_rule_based" in predictions:
        cols = ["go_rule_based"] + cols
        col_labels = ["Go"] + col_labels
    header = f"  {'Никнейм':<35} {'Метка':<10}" + "".join(f"{c:<8}" for c in col_labels)
    print(header)
    print("  " + "-" * (35 + 10 + 8 * len(cols)))

    for i, (text, true_label) in enumerate(zip(all_texts, all_labels)):
        preds_here = [predictions[c][i] for c in cols]
        # Показываем строку, если хотя бы одна система ошиблась.
        if any(p != true_label for p in preds_here):
            label_str = "разрешён" if true_label == 1 else "запрещён"
            marks = []
            for p in preds_here:
                marks.append("+" if p == true_label else "-")
            marks_str = "".join(f"{m:<8}" for m in marks)
            print(f"  {text[:33]:<35} {label_str:<10} {marks_str}")

    # Сохраняем результаты.
    results_df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    _save_bar_chart(results_df)
    _save_confusion_matrices(predictions, all_texts, all_labels)

    print(f"\nРезультаты сохранены в: {RESULTS_DIR}/")

# Графики.
def _save_bar_chart(results_df: pd.DataFrame) -> None:
    """Столбчатая диаграмма F1 по всем системам."""
    valid = results_df[results_df["f1_all"].notna()].copy()
    if valid.empty:
        return

    systems = valid["system"].tolist()
    f1_all = valid["f1_all"].tolist()
    f1_obf = valid["f1_obf"].fillna(0).tolist()

    x = np.arange(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, f1_all, width,
                   label="F1 (все тест-кейсы)", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, f1_obf, width,
                   label="F1 (обфусцированные)", color="#DD8452")

    ax.set_ylabel("F1-score (macro)")
    ax.set_title("Сравнение систем детекции нецензурных никнеймов")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.bar_label(bars1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.2f", padding=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_f1.png", dpi=150)
    plt.close()
    print(f"График: {RESULTS_DIR}/comparison_f1.png")


def _save_confusion_matrices(
    predictions: dict, texts: list[str], labels: list[int]
) -> None:
    """Матрицы ошибок для ML-моделей и Go."""
    show = ["logreg", "svm", "rf"]
    show_labels = ["LogReg", "SVM", "RandomForest"]
    if "go_rule_based" in predictions:
        show = ["go_rule_based"] + show
        show_labels = ["Go (rule-based)"] + show_labels
    n = len(show)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, sys_name, title in zip(axes, show, show_labels):
        if sys_name not in predictions:
            continue
        preds = predictions[sys_name]
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax,
            xticklabels=["запрещён", "разрешён"],
            yticklabels=["запрещён", "разрешён"],
            cmap="Blues",
        )
        ax.set_title(title)
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Истинно")

    plt.suptitle("Матрицы ошибок (тест-кейсы)", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Матрицы ошибок: {RESULTS_DIR}/confusion_matrices.png")


if __name__ == "__main__":
    main()
