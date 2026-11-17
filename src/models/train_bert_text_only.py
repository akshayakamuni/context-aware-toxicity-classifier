"""
E2: BERT-base text-only multi-label classifier for CAD v1.1.

Run from the project root:
    python src/models/train_bert_text_only.py

This experiment uses ONLY the current comment text. Parent/context columns are
intentionally excluded from model inputs.
"""

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "datasets" / "processed"
TRAIN_FILE = DATA_DIR / "cad_train_processed.csv"
DEV_FILE = DATA_DIR / "cad_dev_processed.csv"
TEST_FILE = DATA_DIR / "cad_test_processed.csv"

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
RANDOM_SEED = 42
NUM_WORKERS = 0  # Safe default for Windows.

LABEL_COLUMNS = ["label_0", "label_1", "label_2", "label_3", "label_4"]
LABEL_NAMES = [
    "Neutral",
    "IdentityDirectedAbuse",
    "AffiliationDirectedAbuse",
    "PersonDirectedAbuse",
    "CounterSpeech",
]

MODEL_DIR = PROJECT_ROOT / "models" / "e2_bert_text_only"
REPORT_DIR = PROJECT_ROOT / "reports" / "e2_bert_text_only"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds so runs are as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------

class CADTextDataset(Dataset):
    """Returns one comment and its five multi-label targets."""

    def __init__(self, dataframe: pd.DataFrame):
        self.texts = dataframe["text"].fillna("").astype(str).tolist()
        self.labels = dataframe[LABEL_COLUMNS].astype(np.float32).to_numpy()
        self.ids = dataframe["id"].astype(str).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {
            "text": self.texts[index],
            "labels": torch.tensor(self.labels[index], dtype=torch.float32),
            "id": self.ids[index],
        }


class BERTBatchCollator:
    """Tokenizes a batch and dynamically pads it to that batch's longest item."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [item["text"] for item in batch]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.stack([item["labels"] for item in batch]),
            "ids": [item["id"] for item in batch],
            "texts": texts,
        }


# ---------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------

class BERTMultiLabelClassifier(nn.Module):
    """
    BERT encoder + dropout + five-output linear classification layer.

    The final hidden state at the [CLS] position is used as the comment
    representation. The forward method returns logits, not probabilities.
    """

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, token=False)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(LABEL_COLUMNS))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_vector = outputs.last_hidden_state[:, 0, :]
        cls_vector = self.dropout(cls_vector)
        logits = self.classifier(cls_vector)

        return logits


# ---------------------------------------------------------------------
# 5. Data loading and validation
# ---------------------------------------------------------------------

def load_split(path: Path, split_name: str) -> pd.DataFrame:
    """Load a processed split while preserving literal strings such as 'None'."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find {split_name} file: {path}")

    dataframe = pd.read_csv(path, keep_default_na=False)

    required_columns = ["id", "text"] + LABEL_COLUMNS
    missing = [column for column in required_columns if column not in dataframe.columns]

    if missing:
        raise ValueError(
            f"{split_name} is missing required columns: {missing}\n"
            f"Available columns: {dataframe.columns.tolist()}"
        )

    dataframe["text"] = dataframe["text"].astype(str)
    dataframe[LABEL_COLUMNS] = dataframe[LABEL_COLUMNS].astype(np.float32)

    if not dataframe[LABEL_COLUMNS].isin([0.0, 1.0]).all().all():
        raise ValueError(f"{split_name} contains label values other than 0 and 1.")

    print(f"{split_name}: {len(dataframe):,} records")

    return dataframe


# ---------------------------------------------------------------------
# 6. Evaluation helpers
# ---------------------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    """Calculate aggregate and per-label multi-label classification metrics."""
    per_label = {}

    for index, label_name in enumerate(LABEL_NAMES):
        per_label[label_name] = {
            "precision": float(
                precision_score(y_true[:, index], y_pred[:, index], zero_division=0)
            ),
            "recall": float(
                recall_score(y_true[:, index], y_pred[:, index], zero_division=0)
            ),
            "f1": float(
                f1_score(y_true[:, index], y_pred[:, index], zero_division=0)
            ),
            "positive_support": int(y_true[:, index].sum()),
        }

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "per_label": per_label,
    }


def select_thresholds(y_true, probabilities):
    """Select one threshold per label by maximizing development-set F1."""
    thresholds = []

    for index in range(len(LABEL_NAMES)):
        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in np.arange(0.05, 0.951, 0.01):
            predictions = (probabilities[:, index] >= threshold).astype(int)
            score = f1_score(
                y_true[:, index],
                predictions,
                zero_division=0,
            )

            # On ties, prefer the threshold closest to 0.5.
            if score > best_f1 or (
                np.isclose(score, best_f1)
                and abs(threshold - 0.5) < abs(best_threshold - 0.5)
            ):
                best_f1 = score
                best_threshold = float(threshold)

        thresholds.append(best_threshold)

    return np.array(thresholds, dtype=np.float32)


# ---------------------------------------------------------------------
# 7. Training and prediction
# ---------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0

    for batch_number, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_number % 100 == 0:
            print(
                f"  Batch {batch_number}/{len(loader)} "
                f"| loss={loss.item():.4f}"
            )

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def predict(model, loader):
    """Return gold labels, probabilities, IDs and original comment texts."""
    model.eval()

    all_labels = []
    all_probabilities = []
    all_ids = []
    all_texts = []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probabilities = torch.sigmoid(logits)

        all_labels.append(batch["labels"].cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())
        all_ids.extend(batch["ids"])
        all_texts.extend(batch["texts"])

    return (
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_probabilities, axis=0),
        all_ids,
        all_texts,
    )


def make_prediction_dataframe(ids, texts, y_true, probabilities, thresholds):
    """Build a CSV-friendly table with labels, probabilities and predictions."""
    predictions = (probabilities >= thresholds.reshape(1, -1)).astype(int)

    result = pd.DataFrame({
        "id": ids,
        "text": texts,
    })

    for index, label_name in enumerate(LABEL_NAMES):
        result[f"true_{label_name}"] = y_true[:, index].astype(int)
        result[f"prob_{label_name}"] = probabilities[:, index]
        result[f"pred_{label_name}"] = predictions[:, index]

    return result


# ---------------------------------------------------------------------
# 8. Main experiment
# ---------------------------------------------------------------------

def main():
    set_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("E2: BERT-BASE TEXT-ONLY MULTI-LABEL CLASSIFICATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("Input: current comment only (parent/context fields are not used)")
    print()

    train_df = load_split(TRAIN_FILE, "Train")
    dev_df = load_split(DEV_FILE, "Development")
    test_df = load_split(TEST_FILE, "Test")

    train_ids = set(train_df["id"])
    dev_ids = set(dev_df["id"])
    test_ids = set(test_df["id"])

    if train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids:
        raise ValueError("ID overlap detected between official splits.")

    print("\nTraining positive-label counts:")
    for index, label_name in enumerate(LABEL_NAMES):
        print(f"{label_name:28s}: {int(train_df[LABEL_COLUMNS[index]].sum())}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=False)
    collator = BERTBatchCollator(tokenizer, MAX_LENGTH)

    train_dataset = CADTextDataset(train_df)
    dev_dataset = CADTextDataset(dev_df)
    test_dataset = CADTextDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
    )

    model = BERTMultiLabelClassifier(MODEL_NAME, DROPOUT).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_dev_macro_f1 = -1.0
    best_epoch = 0
    best_state = None
    epoch_history = []

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
        )

        dev_y, dev_probabilities, _, _ = predict(model, dev_loader)
        dev_predictions = (dev_probabilities >= 0.5).astype(int)
        dev_metrics = calculate_metrics(dev_y, dev_predictions)

        elapsed = time.time() - start_time

        print(f"Train loss: {train_loss:.4f}")
        print(f"Development Macro-F1 at 0.5: {dev_metrics['macro_f1']:.4f}")
        print(f"Development Micro-F1 at 0.5: {dev_metrics['micro_f1']:.4f}")
        print(f"Epoch time: {elapsed / 60:.1f} minutes")

        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_macro_f1_threshold_0_5": dev_metrics["macro_f1"],
            "dev_micro_f1_threshold_0_5": dev_metrics["micro_f1"],
            "epoch_seconds": elapsed,
        })

        # Model selection uses development Macro-F1 at the fixed 0.5 threshold.
        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state_dict": best_state,
                    "model_name": MODEL_NAME,
                    "best_epoch": best_epoch,
                    "best_dev_macro_f1_threshold_0_5": best_dev_macro_f1,
                    "max_length": MAX_LENGTH,
                    "label_names": LABEL_NAMES,
                    "random_seed": RANDOM_SEED,
                },
                MODEL_DIR / "best_model.pt",
            )
            print("Saved new best checkpoint.")

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")

    model.load_state_dict(best_state)

    # Select thresholds using development predictions only.
    dev_y, dev_probabilities, dev_ids, dev_texts = predict(model, dev_loader)
    thresholds = select_thresholds(dev_y, dev_probabilities)
    dev_tuned_predictions = (
        dev_probabilities >= thresholds.reshape(1, -1)
    ).astype(int)
    dev_tuned_metrics = calculate_metrics(dev_y, dev_tuned_predictions)

    print("\nDevelopment thresholds selected:")
    for label_name, threshold in zip(LABEL_NAMES, thresholds):
        print(f"{label_name:28s}: {threshold:.2f}")

    # Final test evaluation: thresholds are frozen from development data.
    test_y, test_probabilities, test_ids, test_texts = predict(model, test_loader)
    test_default_predictions = (test_probabilities >= 0.5).astype(int)
    test_tuned_predictions = (
        test_probabilities >= thresholds.reshape(1, -1)
    ).astype(int)

    test_default_metrics = calculate_metrics(test_y, test_default_predictions)
    test_tuned_metrics = calculate_metrics(test_y, test_tuned_predictions)

    dev_predictions_df = make_prediction_dataframe(
        dev_ids, dev_texts, dev_y, dev_probabilities, thresholds
    )
    test_predictions_df = make_prediction_dataframe(
        test_ids, test_texts, test_y, test_probabilities, thresholds
    )

    dev_predictions_df.to_csv(
        REPORT_DIR / "e2_dev_predictions.csv", index=False
    )
    test_predictions_df.to_csv(
        REPORT_DIR / "e2_test_predictions.csv", index=False
    )

    metrics = {
        "experiment": "E2_BERT_TextOnly",
        "input": "current_comment_only",
        "dataset": "CAD_v1.1",
        "model_name": MODEL_NAME,
        "random_seed": RANDOM_SEED,
        "device": str(DEVICE),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs_configured": EPOCHS,
        "best_epoch": best_epoch,
        "best_dev_macro_f1_threshold_0_5": best_dev_macro_f1,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "loss": "BCEWithLogitsLoss",
        "threshold_selection": "per-label F1 maximization on development set",
        "thresholds": {
            name: float(value)
            for name, value in zip(LABEL_NAMES, thresholds)
        },
        "split_sizes": {
            "train": len(train_df),
            "dev": len(dev_df),
            "test": len(test_df),
        },
        "epoch_history": epoch_history,
        "dev_tuned_threshold_metrics": dev_tuned_metrics,
        "test_default_threshold_metrics": test_default_metrics,
        "test_tuned_threshold_metrics": test_tuned_metrics,
    }

    with open(REPORT_DIR / "e2_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with open(REPORT_DIR / "e2_thresholds.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                name: float(value)
                for name, value in zip(LABEL_NAMES, thresholds)
            },
            file,
            indent=4,
        )

    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "max_length": MAX_LENGTH,
                "best_epoch": best_epoch,
                "random_seed": RANDOM_SEED,
                "label_names": LABEL_NAMES,
                "input": "current_comment_only",
            },
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("E2 FINAL TEST RESULTS")
    print("=" * 70)
    print(f"Best epoch: {best_epoch}")
    print("\nDefault threshold (0.5):")
    print(json.dumps(test_default_metrics, indent=2))
    print("\nDevelopment-tuned thresholds:")
    print(json.dumps(test_tuned_metrics, indent=2))

    print("\nSaved model and reports:")
    print(f"Model: {MODEL_DIR}")
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()
