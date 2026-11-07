# Experiment Log

## Project: Context-Aware Toxicity Classifier

This document tracks research progress, experiments, findings, difficulties and decisions.

---

## Entry 1: Dataset Discovery and Selection

**Status:** Completed

**Objective:** Identify a suitable dataset for context-aware toxicity classification.

**Work completed:**

- Investigated available raw datasets.
- Identified Contextual Abuse Dataset (CAD) v1.1.
- Inspected the official dataset archive.
- Examined train, development and test files.
- Inspected combined metadata and codebook.

**Finding:** CAD v1.1 provides contextual abuse labels and conversation metadata suitable for the research objective.

**Decision:** Use official CAD v1.1 split files as the primary supervised dataset.

---

## Entry 2: Official Split and Label Audit

**Status:** Completed

**Objective:** Verify official dataset structure and label distribution.

**Results:**

| Split | Records |
|---|---:|
| Train | 13,584 |
| Dev | 4,526 |
| Test | 5,307 |
| Total | 23,417 |

Five official labels were identified:

- Neutral
- IdentityDirectedAbuse
- AffiliationDirectedAbuse
- PersonDirectedAbuse
- CounterSpeech

Multi-label records:

| Split | Multi-label records |
|---|---:|
| Train | 265 |
| Dev | 82 |
| Test | 89 |
| Total | 436 |

**Finding:** The dataset is highly imbalanced, with Neutral accounting for approximately 81.6% of records.

**Decision:** Preserve multi-label records instead of forcing all records into a single-class format.

---

## Entry 3: Metadata Duplicate and Label Mismatch Audit

**Status:** Completed

**Objective:** Investigate duplicate metadata IDs and differences between metadata annotations and official labels.

**Results:**

- Metadata rows: 27,494.
- Unique metadata IDs: 26,550.
- Repeated-ID groups: 740.
- Extra duplicate rows: 944.
- Official-label mismatches against metadata aggregation: 127.

Seven inspected examples contained metadata `Slur` annotations but official `Neutral` labels.

**Difficulty:** Repeated metadata IDs contained differing annotation values.

**Decision:** Use official split labels as ground truth. Retain metadata annotations for descriptive analysis.

---

## Entry 4: Context Availability and Reconstruction

**Status:** Completed

**Objective:** Determine how much conversational history can be reconstructed.

**Results:**

| Measure | Records |
|---|---:|
| At least one ancestor | 12,835 |
| At least two ancestors | 7,005 |
| At least three ancestors | 3,188 |
| Zero ancestors | 10,582 |

Average ancestor depth: 1.0898.

Maximum observed depth: 13.

**Leakage checks:**

- All official IDs matched metadata.
- Same-split parent reconstruction was enforced.
- No cross-split parent-child links were reported.
- No thread ID mismatches were found.

**Decision:** Use only same-split ancestor text and never use ancestor labels as model inputs.

---

## Entry 5: Context Annotation Analysis

**Status:** Completed

**Objective:** Compare context annotations with actual parent-text availability.

**Finding:** Context annotation and actual context availability are different properties.

Some records without context annotations still have reconstructable parent text.

**Decision:** Maintain separate variables for context annotations and actual conversation history availability.

---

## Entry 6: Label-Wise Context Availability

**Status:** Completed

**Objective:** Investigate whether conversation context availability differs across toxicity labels.

**Results:**

| Label | Records | Mean depth | Context available |
|---|---:|---:|---:|
| Neutral | 19,107 | 1.0866 | 54.525% |
| IdentityDirectedAbuse | 2,305 | 1.0399 | 54.230% |
| AffiliationDirectedAbuse | 1,199 | 1.0517 | 55.880% |
| PersonDirectedAbuse | 1,054 | 1.3083 | 60.721% |
| CounterSpeech | 217 | 1.5806 | 68.664% |

**Findings:**

- CounterSpeech has the highest context availability.
- Person-directed abuse also has relatively high context availability.
- Neutral and identity-directed abuse have similar context availability.
- CounterSpeech has a comparatively small sample size.

**Research implication:** Context availability may be associated with the target label. This should be considered when evaluating model performance.

**Decision:** Report results separately for context-available and context-unavailable subsets.

---

## Entry 7: Planned Context Experiments

**Status:** Planned

**Objective:** Determine whether conversational history improves toxicity classification.

Four configurations will be compared:

| Experiment | Input |
|---|---|
| A | Current comment only |
| B | Current comment + immediate parent |
| C | Current comment + up to two ancestors |
| D | Current comment + up to three ancestors |

**Experimental controls:**

- Identical train/dev/test splits.
- Same target records across configurations.
- Same-split ancestor text only.
- No ancestor labels as input.
- Explicit handling of unavailable context.
- Validation-only model and threshold selection.

**Evaluation metrics:**

- Precision
- Recall
- F1-score
- Macro-F1
- Micro-F1
- Per-label performance

These experiments have not yet been executed.

---

## Entry 8: Preprocessing

**Status:** Not started

The following tasks are pending:

1. Inspect text quality and formatting.
2. Decide text normalization rules.
3. Handle URLs, usernames, punctuation and emojis.
4. Design multi-label target encoding.
5. Define missing-context representation.
6. Select tokenizer and maximum sequence length.
7. Implement leakage-safe preprocessing.
8. Validate processed data.

Preprocessing statistics and vocabulary must be fitted using training data only wherever applicable.

---

## Entry 9: Model Development

**Status:** Not started

Planned after preprocessing:

- Establish a text-only baseline.
- Implement context-aware configurations.
- Compare model performance.
- Investigate class imbalance handling.
- Perform error analysis.
- Evaluate the contribution of conversational context.

---

## Future Experiment Recording Template

For every new experiment, record:

**Experiment ID:**

**Date:**

**Objective:**

**Dataset and split:**

**Input configuration:**

**Preprocessing:**

**Model:**

**Hyperparameters:**

**Random seed:**

**Evaluation metrics:**

**Results:**

**Observations:**

**Problems encountered:**

**Research decision:**

**Next step:**

---

## Current Project Status

**Completed:** Dataset discovery, official split verification, label audit, duplicate analysis, metadata mismatch analysis, context reconstruction, conversation-depth analysis and label-wise context availability analysis.

**Current stage:** Dataset audit completed.

**Next stage:** Data preprocessing and preparation for controlled context experiments.