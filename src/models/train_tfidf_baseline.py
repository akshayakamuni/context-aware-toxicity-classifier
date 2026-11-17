"""E1: TF-IDF + One-vs-Rest Logistic Regression (current comment only)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, hamming_loss, precision_score, recall_score
)
from sklearn.multiclass import OneVsRestClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "processed"
ARTIFACTS = ROOT / "models" / "e1_tfidf_logreg"
REPORTS = ROOT / "reports" / "e1_tfidf_logreg"

LABELS = [
    "Neutral", "IdentityDirectedAbuse", "AffiliationDirectedAbuse",
    "PersonDirectedAbuse", "CounterSpeech"
]
LABEL_COLS = [f"label_{i}" for i in range(5)]
EXPECTED = {"train": 13584, "dev": 4526, "test": 5307}
SEED = 42
THRESHOLD_GRID = np.arange(0.05, 0.951, 0.01)


def load_split(filename: str, split: str) -> pd.DataFrame:
    # keep_default_na=False preserves literal "None" and "NA" text.
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path, keep_default_na=False)

    required = {"id", "text", "split", *LABEL_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename}: missing columns {sorted(missing)}")
    if len(df) != EXPECTED[split]:
        raise ValueError(f"{split}: expected {EXPECTED[split]} rows, got {len(df)}")
    if not (df["split"] == split).all():
        raise ValueError(f"{filename} contains an incorrect split value.")
    if df["id"].duplicated().any():
        raise ValueError(f"Duplicate IDs in {filename}.")

    df["text"] = df["text"].astype(str)
    for col in LABEL_COLS:
        values = pd.to_numeric(df[col], errors="raise")
        if not values.isin([0, 1]).all():
            raise ValueError(f"{col} must contain only 0/1.")
        df[col] = values.astype(np.int32)
    return df


def targets(df: pd.DataFrame) -> np.ndarray:
    return df[LABEL_COLS].to_numpy(dtype=np.int32)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
    }


def per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    result = {}
    for i, label in enumerate(LABELS):
        result[label] = {
            "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "positive_support": int(y_true[:, i].sum()),
        }
    return result


def tune_thresholds(y_dev: np.ndarray, p_dev: np.ndarray) -> list[float]:
    # One threshold per label, selected only from development data.
    chosen = []
    for i, label in enumerate(LABELS):
        best_t, best_f1 = 0.50, -1.0
        for t in THRESHOLD_GRID:
            pred = (p_dev[:, i] >= t).astype(np.int32)
            score = f1_score(y_dev[:, i], pred, zero_division=0)
            if score > best_f1 or (
                np.isclose(score, best_f1) and abs(t - 0.50) < abs(best_t - 0.50)
            ):
                best_t, best_f1 = float(t), float(score)
        chosen.append(best_t)
        print(f"{label:28s} threshold={best_t:.2f} dev_F1={best_f1:.4f}")
    return chosen


def apply_thresholds(probabilities: np.ndarray, thresholds: list[float]) -> np.ndarray:
    return (probabilities >= np.asarray(thresholds).reshape(1, -1)).astype(np.int32)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_predictions(
    df: pd.DataFrame, probs: np.ndarray, preds: np.ndarray, path: Path
) -> None:
    out = pd.DataFrame({"id": df["id"], "text": df["text"]})
    y = targets(df)
    for i, label in enumerate(LABELS):
        out[f"true_{label}"] = y[:, i]
        out[f"prob_{label}"] = probs[:, i]
        out[f"pred_{label}"] = preds[:, i]
    out.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("E1: TF-IDF + ONE-VS-REST LOGISTIC REGRESSION")
    print("Input: current comment only (no parent/context fields)")
    print("=" * 68)

    train = load_split("cad_train_processed.csv", "train")
    dev = load_split("cad_dev_processed.csv", "dev")
    test = load_split("cad_test_processed.csv", "test")

    # Check that official record IDs are disjoint across splits.
    ids = [set(train["id"]), set(dev["id"]), set(test["id"])]
    if (ids[0] & ids[1]) or (ids[0] & ids[2]) or (ids[1] & ids[2]):
        raise ValueError("Record ID overlap detected between official splits.")

    print(f"Train: {len(train)} | Dev: {len(dev)} | Test: {len(test)}")
    print("Cross-split ID overlap: 0")

    # Deliberately select only text and labels. Parent fields are not read.
    x_train_text = train["text"].tolist()
    x_dev_text = dev["text"].tolist()
    x_test_text = test["text"].tolist()
    y_train, y_dev, y_test = targets(train), targets(dev), targets(test)

    print("\nTraining positive-label counts:")
    for i, label in enumerate(LABELS):
        print(f"{label:28s}: {int(y_train[:, i].sum())}")

    print("\nFitting TF-IDF on training text only...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=200000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        dtype=np.float64,
    )
    x_train = vectorizer.fit_transform(x_train_text)
    x_dev = vectorizer.transform(x_dev_text)
    x_test = vectorizer.transform(x_test_text)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"Matrix shapes: train={x_train.shape}, dev={x_dev.shape}, test={x_test.shape}")

    print("\nTraining five one-vs-rest Logistic Regression classifiers...")
    base = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )
    model = OneVsRestClassifier(base, n_jobs=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)
    for warning in caught:
        print("WARNING:", warning.message)

    print("\nPredicting development and test probabilities...")
    p_dev = model.predict_proba(x_dev)
    p_test = model.predict_proba(x_test)

    print("\nSelecting one threshold per label using DEV only...")
    tuned = tune_thresholds(y_dev, p_dev)
    default = [0.50] * len(LABELS)

    dev_default = apply_thresholds(p_dev, default)
    dev_tuned = apply_thresholds(p_dev, tuned)
    test_default = apply_thresholds(p_test, default)
    test_tuned = apply_thresholds(p_test, tuned)

    result = {
        "experiment": "E1_TFIDF_LogisticRegression",
        "input": "current_comment_only",
        "dataset": "CAD_v1.1",
        "random_seed": SEED,
        "scikit_learn_version": sklearn.__version__,
        "split_sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
        "tfidf": {
            "ngram_range": [1, 2], "min_df": 2, "max_features": 200000,
            "sublinear_tf": True, "strip_accents": "unicode", "lowercase": True
        },
        "logistic_regression": {
            "C": 1.0, "solver": "liblinear", "max_iter": 2000,
            "class_weight": "balanced"
        },
        "vocabulary_size": len(vectorizer.vocabulary_),
        "threshold_selection": "per-label F1 maximization on development set",
        "thresholds": {label: tuned[i] for i, label in enumerate(LABELS)},
        "dev_default_threshold_metrics": metrics(y_dev, dev_default),
        "dev_default_threshold_per_label": per_label_metrics(y_dev, dev_default),
        "dev_tuned_threshold_metrics": metrics(y_dev, dev_tuned),
        "dev_tuned_threshold_per_label": per_label_metrics(y_dev, dev_tuned),
        "test_default_threshold_metrics": metrics(y_test, test_default),
        "test_default_threshold_per_label": per_label_metrics(y_test, test_default),
        "test_tuned_threshold_metrics": metrics(y_test, test_tuned),
        "test_tuned_threshold_per_label": per_label_metrics(y_test, test_tuned),
    }

    save_json(result, REPORTS / "e1_metrics.json")
    save_json(result["thresholds"], REPORTS / "e1_thresholds.json")
    save_predictions(dev, p_dev, dev_tuned, REPORTS / "e1_dev_predictions.csv")
    save_predictions(test, p_test, test_tuned, REPORTS / "e1_test_predictions.csv")

    joblib.dump(vectorizer, ARTIFACTS / "tfidf_vectorizer.joblib")
    joblib.dump(model, ARTIFACTS / "ovr_logistic_regression.joblib")
    save_json({
        "experiment": "E1",
        "model": "TF-IDF + OneVsRest Logistic Regression",
        "input_field": "text",
        "excluded_fields": ["parent_text", "parent_id", "context_available", "parent_status"],
        "labels": LABELS,
        "random_seed": SEED,
        "scikit_learn_version": sklearn.__version__,
    }, ARTIFACTS / "model_metadata.json")

    print("\n" + "=" * 68)
    print("E1 TEST RESULTS")
    print("=" * 68)
    for heading, key in [
        ("Default threshold = 0.50", "test_default_threshold_metrics"),
        ("Dev-tuned thresholds", "test_tuned_threshold_metrics"),
    ]:
        print(f"\n{heading}:")
        for name, value in result[key].items():
            print(f"{name:22s}: {value:.4f}")

    print("\nPer-label F1 (dev-tuned thresholds):")
    for label in LABELS:
        print(f"{label:28s}: {result['test_tuned_threshold_per_label'][label]['f1']:.4f}")

    print(f"\nModel artifacts: {ARTIFACTS}")
    print(f"Reports:         {REPORTS}")
    print("\nE1 completed successfully.")


if __name__ == "__main__":
    main()
