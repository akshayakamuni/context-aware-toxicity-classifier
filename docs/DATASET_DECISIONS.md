# Dataset Decisions

## Project: Context-Aware Toxicity Classifier

This document records the decisions made during dataset investigation and auditing. These decisions will guide preprocessing, model development and evaluation.

---

## 1. Dataset Selection

| Decision | Final choice | Reason |
|---|---|---|
| Primary dataset | CAD v1.1 | Official contextual abuse dataset |
| Dataset source | Official train/dev/test TSV files | Official splits validated |
| Dataset version | v1.1 | Corrected release |
| Total records | 23,417 | Verified |
| Metadata matching | `info_id` | All official IDs matched successfully |

## 2. Label Decisions

The five official labels are:

1. Neutral
2. IdentityDirectedAbuse
3. AffiliationDirectedAbuse
4. PersonDirectedAbuse
5. CounterSpeech

Decisions:

- Use the five labels present in the official split files.
- Do not create a separate `Slur` output class from metadata annotations.
- Preserve all 436 multi-label records.
- Use official split TSV labels as ground truth.
- Do not silently modify labels based on metadata annotations.

## 3. Duplicate Handling

- Metadata contains 27,494 rows and 26,550 unique IDs.
- Repeated metadata IDs must not be treated as separate training examples.
- Use one metadata entry per `info_id` for context reconstruction.
- Preserve annotation differences for research analysis.
- Do not blindly aggregate conflicting annotations into training labels.

## 4. Context Reconstruction Decisions

A parent comment is considered usable only when:

- Its ID exists in metadata.
- It belongs to the same official split as the target comment.
- Its text is non-empty.

Additional decisions:

- Use only same-split ancestor text.
- Never use ancestor labels as model inputs.
- Do not fabricate missing conversation text.
- Preserve target records even when context is unavailable.
- Keep context annotations separate from actual context availability.

## 5. Context Experiment Configurations

We will compare four configurations:

| Configuration | Input |
|---|---|
| A | Current comment only |
| B | Current comment + immediate parent |
| C | Current comment + up to two ancestors |
| D | Current comment + up to three ancestors |

The purpose is to determine whether additional conversational history improves toxicity classification.

We will not assume that more context always improves performance.

## 6. Evaluation Decisions

Because the dataset is highly imbalanced:

- Accuracy will not be the only evaluation metric.
- Report per-class precision, recall and F1-score.
- Report macro-F1 and micro-F1.
- Evaluate context-available and context-unavailable records separately.
- Select classification thresholds using validation data only.
- Keep the official test set untouched until final evaluation.

## 7. Important Dataset Observations

| Observation | Value |
|---|---:|
| Official records | 23,417 |
| Multi-label records | 436 |
| Neutral records | 19,107 |
| CounterSpeech records | 217 |
| Records with at least one ancestor | 12,835 |
| Records with at least two ancestors | 7,005 |
| Records with at least three ancestors | 3,188 |
| Average ancestor depth | 1.0898 |
| Maximum ancestor depth | 13 |

CounterSpeech has the highest observed context availability (68.664%), while Neutral has 54.525%.

These are descriptive findings, not proof that context improves classification.

## 8. Decisions Still Pending

The following decisions will be made during preprocessing and model development:

1. Multi-label target encoding and loss function.
2. Handling of class imbalance.
3. Missing-context representation.
4. Tokenizer and maximum sequence length.
5. Text normalization rules.
6. Model architecture.
7. Threshold selection strategy.

Each decision must be documented with its reasoning and experimental evidence.