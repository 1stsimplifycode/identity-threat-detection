"""Fine-tuned Hugging Face detector (Phase 3) -- `prajjwal1/bert-tiny`
(~4.4M params), genuinely fine-tuned on THIS project's own engineered
feature sequences (not a frozen inference call), evaluated on the same
held-out test split and the same metrics as every other model in the
comparison table. Loaded once from the free, open-weight HF Hub checkpoint
-- no paid inference endpoint involved.

Honest framing (documented, not hidden -- see docs/phase_3_report.md): a
tiny transformer over already-engineered TABULAR features is a harder
learning problem than a tree model for the same information; XGBoost has
the right inductive bias for this kind of structured, low-dimensional data.
The one place a sequence encoder can plausibly add value a single-row
XGBoost input can't is modeling the ORDER in which signals accumulate
across a user's recent events -- which is why each training example is a
short pseudo-text SEQUENCE of the last k serialized events, not a single
row -- verified empirically against the shared metrics table, not assumed
to help in advance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizerFast

from feature_engineering.pipeline import FEATURE_COLUMNS
from models.xgboost_classifier import ATTACK_TYPE_CLASSES, encode_labels, multiclass_labels


def subsample_for_training(sequences: pd.DataFrame, train_labels: pd.DataFrame, cfg: DictConfig, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ALL attack-labeled train rows are kept; the benign majority is
    subsampled to at most `max_benign_train_examples` -- see
    configs/models/default.yaml for why (CPU-only fine-tuning throughput).
    Returns the (possibly shrunk) `train_labels` and matching `sequences`.
    """
    max_benign = int(cfg.models.hf_classifier.max_benign_train_examples)
    rng = np.random.default_rng(seed)

    is_attack = train_labels["attack_type"].notna()
    attack_ids = train_labels.loc[is_attack, "record_id"]
    benign_ids = train_labels.loc[~is_attack, "record_id"]

    if len(benign_ids) > max_benign:
        benign_ids = pd.Series(rng.choice(benign_ids.to_numpy(), size=max_benign, replace=False))

    kept_ids = set(attack_ids) | set(benign_ids)
    kept_labels = train_labels[train_labels["record_id"].isin(kept_ids)]
    kept_sequences = sequences[sequences["record_id"].isin(kept_ids)]
    return kept_sequences, kept_labels


def _serialize_event(row: dict) -> str:
    parts = []
    for col in FEATURE_COLUMNS:
        val = row[col]
        val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
        parts.append(f"{col}={val_str}")
    return " ".join(parts)


def build_sequences(features: pd.DataFrame, events: pd.DataFrame, k: int) -> pd.DataFrame:
    """One row per event: record_id + a short pseudo-text sequence
    concatenating this event's serialized features with up to k-1
    preceding events from the SAME user (chronological), joined by ' | '.
    """
    merged = events[["record_id", "user_id", "timestamp"]].merge(features, on="record_id")
    merged = merged.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    rows: list[dict] = []
    for _, group in merged.groupby("user_id", sort=False):
        group = group.reset_index(drop=True)
        single_texts = [_serialize_event(r) for r in group.to_dict("records")]
        for i in range(len(group)):
            window = single_texts[max(0, i - k + 1): i + 1]
            rows.append({"record_id": group.loc[i, "record_id"], "sequence_text": " | ".join(window)})
    return pd.DataFrame(rows)


class _SequenceDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None, tokenizer: BertTokenizerFast, max_length: int):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        self.labels = labels

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_hf_classifier(sequences: pd.DataFrame, train_labels: pd.DataFrame, cfg: DictConfig) -> tuple[BertForSequenceClassification, BertTokenizerFast]:
    hf_cfg = cfg.models.hf_classifier
    # BertTokenizerFast explicitly, not AutoTokenizer -- found by direct
    # testing that AutoTokenizer misidentifies the tokenizer class for this
    # (older, lightly-maintained) checkpoint's repo layout and fails with a
    # spurious "need sentencepiece" error; BertTokenizerFast loads it fine.
    tokenizer = BertTokenizerFast.from_pretrained(str(hf_cfg.checkpoint))
    model = BertForSequenceClassification.from_pretrained(str(hf_cfg.checkpoint), num_labels=len(ATTACK_TYPE_CLASSES))

    label_frame = pd.DataFrame({
        "record_id": train_labels["record_id"].to_numpy(),
        "label_index": encode_labels(multiclass_labels(train_labels)),
    })
    merged = sequences.merge(label_frame, on="record_id")

    dataset = _SequenceDataset(
        merged["sequence_text"].tolist(), merged["label_index"].tolist(), tokenizer, int(hf_cfg.max_length),
    )
    loader = DataLoader(dataset, batch_size=int(hf_cfg.batch_size), shuffle=True)

    device = torch.device("cpu")  # CPU-only by design -- Render free-tier has no GPU
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(hf_cfg.learning_rate))

    model.train()
    for epoch in range(int(hf_cfg.epochs)):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += float(outputs.loss.item())
        print(f"[hf_classifier] epoch {epoch + 1}/{int(hf_cfg.epochs)} mean loss={total_loss / len(loader):.4f}")

    return model, tokenizer


def score_hf_classifier(model: BertForSequenceClassification, tokenizer: BertTokenizerFast, sequences: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    hf_cfg = cfg.models.hf_classifier
    model.eval()
    device = torch.device("cpu")
    benign_index = ATTACK_TYPE_CLASSES.index("benign")
    batch_size = int(hf_cfg.batch_size)

    texts = sequences["sequence_text"].tolist()
    record_ids = sequences["record_id"].tolist()

    scores: list[float] = []
    predicted_indices: list[int] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=int(hf_cfg.max_length), return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            scores.extend((1.0 - proba[:, benign_index]).tolist())
            predicted_indices.extend(proba.argmax(axis=1).tolist())

    predicted_class = [ATTACK_TYPE_CLASSES[i] for i in predicted_indices]
    return pd.DataFrame({"record_id": record_ids, "hf_anomaly_score": scores, "hf_predicted_class": predicted_class})
