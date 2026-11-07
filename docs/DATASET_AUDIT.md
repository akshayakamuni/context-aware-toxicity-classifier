# Dataset Audit Report

## Project: Context-Aware Toxicity Classifier

**Primary Dataset:** Contextual Abuse Dataset (CAD) v1.1

**Objective:** Investigate dataset quality, official labels, class distribution, metadata consistency, conversation history, context availability and potential data leakage before preprocessing and model training.

---

## 1. Dataset Provenance

The primary dataset selected for this project is the Contextual Abuse Dataset (CAD), introduced by Vidgen et al. in *Introducing CAD: The Contextual Abuse Dataset* (NAACL 2021).

- Research paper: https://www.aclweb.org/anthology/2021.naacl-main.182/
- Dataset DOI: https://doi.org/10.5281/zenodo.4881008
- GitHub repository: https://github.com/dongpng/cad_naacl2021

The project uses the official CAD v1.1 train, development and test files.

CAD primarily contains English Reddit comments and provides annotations for contextual abuse categories.

Version 1.1 includes corrections to primary and secondary category annotations for 95 entries.

The archive contains:
- Official v1.1 train, dev and test TSV files.
- Combined metadata TSV.
- Earlier CAD v1 files.
- Dataset codebook.
- README and errata.

CAD v1 is retained for reference. CAD v1.1 is the primary dataset for this project.

## 2. Raw Dataset Inventory

The following files and datasets were identified in `datasets/raw/`:

| File/Dataset | Description |
|---|---|
| `data.zip` | Primary CAD archive |
| `CADD.json` | Additional CADD dataset |
| `CADD_train.json` | CADD training data |
| `CADD_dev.json` | CADD development data |
| `CADD_test.json` | CADD testing data |
| `Multi-Sarcasm.csv` | Sarcasm dataset |
| `unhealthy_full.csv` | Additional toxicity-related dataset |
| `unhealthy-conversations-main.zip` | Conversation dataset archive |
| Hindi hate-speech CSV files | Hindi-language data |
| `hi_Hasoc2021_train.zip` | Password-protected dataset |
| Parquet shards | Additional train, validation and test data |
| `youtoxic_english_1000.csv` | English toxicity dataset |
| `abalone.csv` | Unrelated dataset |

The password-protected HASOC archive was deliberately excluded rather than decrypted.

The CADD JSON files were not selected as the primary supervised source because their numeric label mapping was not established during this audit.

Other datasets and parquet shards have not yet been incorporated into the primary CAD experiments.

Raw dataset files were preserved unchanged.

## 3. Official Dataset Structure

The official CAD v1.1 split files contain three main columns:

- `id`
- `text`
- `labels`

### Official split sizes

| Split | Records |
|---|---:|
| Train | 13,584 |
| Dev | 4,526 |
| Test | 5,307 |
| **Total** | **23,417** |

All official split counts were validated successfully.

Official IDs are unique across the three split files.

## 4. Official Labels

The official split files contain five primary labels:

1. Neutral
2. IdentityDirectedAbuse
3. AffiliationDirectedAbuse
4. PersonDirectedAbuse
5. CounterSpeech

The codebook also discusses non-hateful slur and reclaimed uses. However, `Slur` is not a separate primary output in the official v1.1 split files.

Therefore, the initial supervised task uses the five official labels only.

### 4.1 Label Distribution

| Label | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Neutral | 11,073 | 3,692 | 4,342 | 19,107 |
| IdentityDirectedAbuse | 1,357 | 434 | 514 | 2,305 |
| AffiliationDirectedAbuse | 711 | 246 | 242 | 1,199 |
| PersonDirectedAbuse | 612 | 205 | 237 | 1,054 |
| CounterSpeech | 113 | 38 | 66 | 217 |
| **Total** | **13,584** | **4,526** | **5,307** | **23,417** |

### 4.2 Class Imbalance

Neutral accounts for approximately 81.6% of all official records.

CounterSpeech accounts for approximately 0.93%.

This indicates significant class imbalance.

Accuracy alone will not be sufficient to evaluate the final classifier.

### 4.3 Multi-Label Records

| Split | Multi-label records |
|---|---:|
| Train | 265 |
| Dev | 82 |
| Test | 89 |
| **Total** | **436** |

The official dataset contains 436 multi-label records.

Therefore, the task should preserve multi-label annotations rather than automatically converting every record into one mutually exclusive class.

The final target encoding and loss function will be decided during model design.

## 5. Combined Metadata Structure

The combined metadata file is `cad_v1_1.tsv`.

It contains:

- 27,494 rows.
- 23 columns.
- 26,550 unique metadata IDs.

Important columns include:

- `info_id`
- `info_id.parent`
- `info_id.link`
- `info_thread.id`
- `info_order`
- `annotation_Primary`
- `annotation_Secondary`
- `annotation_Context`
- `annotation_Target`
- `annotation_Target_top.level.category`
- `meta_text`
- `split`

### Metadata split counts

| Split/category | Rows |
|---|---:|
| Train | 14,113 |
| Test | 5,495 |
| Dev | 4,684 |
| exclude_empty | 2,244 |
| exclude_bot | 604 |
| exclude_lang | 353 |
| exclude_image | 1 |
| **Total** | **27,494** |

An important finding was that official split IDs must be matched to the metadata using `info_id`, not the metadata row field `id`.

The corrected mapping successfully matched all 23,417 official records.

## 6. Duplicate Metadata Audit

The metadata contains:

- 27,494 raw rows.
- 26,550 unique IDs.
- 740 repeated-ID groups.
- 944 extra duplicate rows.

The duplicate audit identified differing annotation values among repeated IDs.

| Field | Repeated-ID groups with differing values |
|---|---:|
| Primary annotations | 480 |
| Secondary annotations | 591 |
| Context annotations | 139 |
| Text | 0 |
| Parent ID | 0 |
| Split | 0 |
| Thread ID | 0 |
| Order | 0 |

Repeated metadata rows must not be treated as separate supervised examples.

For context reconstruction, one metadata entry per `info_id` is used after checking parent and text consistency.

## 7. Metadata Labels Versus Official Labels

Aggregating metadata primary annotations by unique ID produced:

| Category | Records |
|---|---:|
| Neutral | 21,935 |
| IdentityDirectedAbuse | 2,425 |
| AffiliationDirectedAbuse | 1,236 |
| PersonDirectedAbuse | 1,110 |
| CounterSpeech | 220 |
| Slur | 140 |

These are metadata aggregation results, not the official supervised-label distribution.

Comparing aggregated metadata Primary annotations with official split labels produced 127 mismatches.

| Split | Mismatches |
|---|---:|
| Train | 77 |
| Dev | 31 |
| Test | 19 |
| **Total** | **127** |

Seven inspected mismatches had metadata `Slur` but official `Neutral` labels.

The inspected examples appeared to discuss offensive language rather than use it abusively.

The exact reason for all 127 mismatches has not been established.

**Decision:** Official split TSV labels are the source of truth for supervised learning. Metadata annotations are retained for descriptive analysis. No labels are silently modified.

## 8. Context Availability Audit

Conversation context was reconstructed using:

`info_id.parent`

A parent was considered usable only when:

1. Its ID existed in metadata.
2. It belonged to the same official train/dev/test split as the target.
3. It had non-empty text.

### 8.1 Parent availability by split

| Split | No parent ID | Parent missing metadata | Same-split parent available | Parent excluded from official splits | Parent text missing |
|---|---:|---:|---:|---:|---:|
| Train | 880 | 4,880 | 7,570 | 251 | 3 |
| Dev | 357 | 1,712 | 2,390 | 66 | 1 |
| Test | 423 | 1,930 | 2,875 | 79 | 0 |
| **Total** | **1,660** | **8,522** | **12,835** | **396** | **4** |

A total of 12,835 records have at least one usable same-split parent.

This represents approximately 54.81% of the official dataset.

The earlier context reconstruction audit found no cross-split parent-child relationships.

## 9. Conversation Depth Audit

The corrected conversation-depth audit uses unique official IDs rather than iterating over all metadata rows.

It validated all 23,417 official records.

### 9.1 Conversation depth distribution

| Ancestor depth | Records | Percentage |
|---:|---:|---:|
| 0 | 10,582 | 45.19% |
| 1 | 5,830 | 24.90% |
| 2 | 3,817 | 16.30% |
| 3 | 1,733 | 7.40% |
| 4 | 804 | 3.43% |
| 5 | 513 | 2.19% |
| 6 | 48 | 0.20% |
| 7 | 27 | 0.12% |
| 8 | 23 | 0.10% |
| 9 | 14 | 0.06% |
| 10 | 10 | 0.04% |
| 11 | 8 | 0.03% |
| 12 | 4 | 0.02% |
| 13 | 4 | 0.02% |
| **Total** | **23,417** | **100%** |

### 9.2 Important findings

- Average ancestor depth: 1.0898.
- At least one ancestor: 12,835 records (54.81%).
- At least two ancestors: 7,005 records (29.91%).
- At least three ancestors: 3,188 records (13.62%).
- Maximum observed reconstructable depth: 13.

### 9.3 Chain termination reasons

| Termination reason | Records |
|---|---:|
| Parent missing from metadata | 21,032 |
| Root or no parent | 1,660 |
| Parent excluded from official splits | 720 |
| Parent text missing | 5 |
| **Total** | **23,417** |

Termination reason indicates why reconstruction eventually stopped.

For example, a conversation may have two usable ancestors and then stop because the next parent is missing.

Therefore, 21,032 missing-parent terminations does not mean 21,032 records have zero context.

## 10. Thread Consistency

The conversation audit reported:

- Unique threads: 1,395.
- Parent-child relationships checked: 14,062.
- Relationships with missing thread ID: 0.
- Parent-child thread ID mismatches: 0.

No cross-split parent-child relationships or cycles were reported by the corrected conversation-depth audit.

Split validation must still be enforced in future context construction.

## 11. Context Annotation Versus Actual Availability

The metadata contains context annotations such as:

- `CurrentContent`
- `PreviousContent`

These annotations do not guarantee that actual parent text is available.

### Context annotation versus parent availability

| Context annotation | Parent available | Parent unavailable |
|---|---:|---:|
| CurrentContent | 1,569 | 1,365 |
| CurrentContent + PreviousContent | 84 | 51 |
| No context annotation | 10,357 | 8,642 |
| PreviousContent | 825 | 524 |

A record without a context annotation can still have reconstructable parent text.

Similarly, an annotation does not guarantee that parent text is available.

These variables must remain separate.

## 12. Label-Wise Context Availability Audit

A separate audit compared conversation depth across the five official labels.

All 23,417 official records were successfully matched to metadata through `info_id`.

### 12.1 Context availability by label

| Label | Records | Mean depth | At least 1 ancestor | At least 2 ancestors | At least 3 ancestors |
|---|---:|---:|---:|---:|---:|
| Neutral | 19,107 | 1.0866 | 54.525% | 29.743% | 13.545% |
| IdentityDirectedAbuse | 2,305 | 1.0399 | 54.230% | 28.764% | 12.668% |
| AffiliationDirectedAbuse | 1,199 | 1.0517 | 55.880% | 29.274% | 12.510% |
| PersonDirectedAbuse | 1,054 | 1.3083 | 60.721% | 36.243% | 18.786% |
| CounterSpeech | 217 | 1.5806 | 68.664% | 48.848% | 23.502% |

### 12.2 Findings

- CounterSpeech has the highest observed context availability.
- CounterSpeech also has the highest mean ancestor depth.
- Person-directed abuse has higher context availability than Neutral and the other two abuse labels.
- Neutral and identity-directed abuse have similar context availability.
- CounterSpeech has only 217 records, so its percentages must be interpreted cautiously.

These findings are descriptive and do not prove that context improves classification.

### 12.3 Context annotation distribution by label

| Label | Context annotation | Records | Percentage within label |
|---|---|---:|---:|
| Neutral | NoAnnotation | 18,999 | 99.435% |
| Neutral | CurrentOnly | 108 | 0.565% |
| IdentityDirectedAbuse | CurrentOnly | 1,599 | 69.371% |
| IdentityDirectedAbuse | PreviousOnly | 603 | 26.161% |
| IdentityDirectedAbuse | CurrentAndPrevious | 103 | 4.469% |
| AffiliationDirectedAbuse | CurrentOnly | 860 | 71.726% |
| AffiliationDirectedAbuse | PreviousOnly | 291 | 24.270% |
| AffiliationDirectedAbuse | CurrentAndPrevious | 48 | 4.003% |
| PersonDirectedAbuse | CurrentOnly | 688 | 65.275% |
| PersonDirectedAbuse | PreviousOnly | 286 | 27.135% |
| PersonDirectedAbuse | CurrentAndPrevious | 80 | 7.590% |
| CounterSpeech | PreviousOnly | 194 | 89.401% |
| CounterSpeech | CurrentAndPrevious | 23 | 10.599% |

The annotation distribution is strongly associated with the labels in this dataset.

These annotation categories should not automatically be used as model features because they may expose annotation-process information associated with the target.

## 13. Dataset Challenges Identified

1. **Class imbalance:** Neutral dominates the dataset.
2. **Rare CounterSpeech category:** Only 217 records are available.
3. **Multi-label targets:** 436 records contain multiple labels.
4. **Duplicate metadata:** Repeated IDs have differing annotation rows.
5. **Label mismatches:** 127 metadata-aggregated labels differ from official split labels.
6. **Incomplete context:** Only 54.81% have at least one reconstructable ancestor.
7. **Unequal context availability:** Context availability differs across labels.
8. **Annotation differences:** Context annotation and actual parent-text availability are not equivalent.
9. **External datasets:** Other raw datasets have not yet been incorporated into the primary CAD experiments.
10. **No causal evidence:** Dataset audits alone cannot establish whether context improves model performance.

## 14. Planned Context Experiments

Four configurations will be compared.

| Configuration | Input |
|---|---|
| A | Current comment only |
| B | Current comment + immediate parent |
| C | Current comment + up to two ancestors |
| D | Current comment + up to three ancestors |

Experimental controls:

- Use identical official train/dev/test splits.
- Preserve all target records.
- Use only same-split ancestor text.
- Never use ancestor labels as model inputs.
- Do not fabricate missing text.
- Represent unavailable context explicitly.
- Evaluate context-available and context-unavailable subsets separately.
- Report per-label precision, recall and F1.
- Report macro-F1 and micro-F1.
- Do not rely on accuracy alone.
- Select thresholds using validation data only.

## 15. Audit Completion Status

Completed:

- Dataset provenance and archive integrity.
- Official split and ID verification.
- Official label distribution.
- Multi-label record analysis.
- Class imbalance analysis.
- Duplicate metadata analysis.
- Metadata versus official-label mismatch analysis.
- Parent availability analysis.
- Leakage-safe context reconstruction.
- Conversation depth analysis.
- Thread consistency checks.
- Context annotation analysis.
- Label-wise context availability and depth analysis.

Not yet completed:

- Text preprocessing.
- Tokenization.
- Vocabulary construction.
- Model training.
- Hyperparameter tuning.
- Statistical significance testing of label-wise context differences.
- Investigation of every other raw dataset.

## Conclusion

The official CAD v1.1 split files have been validated as the primary supervised dataset for this project.

The audit identified strong class imbalance, multi-label targets, incomplete conversation histories and label-wise differences in context availability.

These findings will guide preprocessing and controlled experiments comparing different conversation-context lengths.




# Dataset Audit and Exploration Report

## Project: Context-Aware Toxicity Classifier

**Dataset:** Contextual Abuse Dataset (CAD) v1.1  
**Dataset type:** Context-dependent abuse classification  
**Primary language:** English  
**Primary source:** CAD dataset released with the NAACL 2021 paper  
**Audit status:** Initial dataset investigation completed  
**Training status:** Not started

---

# 1. Project Objective

The objective of this project is to develop a context-aware toxicity classification system capable of distinguishing abusive language from neutral discussions, counter-speech and other context-dependent language.

A conventional toxicity classifier may identify offensive words without understanding how they are used.

For example:

- A person directly insulting someone may be abusive.
- A person quoting an offensive expression to condemn it may be expressing counter-speech.
- A person discussing an identity group without attacking it may be neutral.

Therefore, the project investigates how conversational context can improve classification performance.

The CAD dataset was selected because it contains contextual abuse annotations, conversation information and multiple abuse-related categories.

# 2. Dataset Source and Provenance

The primary dataset used in this project is the Contextual Abuse Dataset (CAD), version 1.1.

**Research paper:**

Vidgen et al., Introducing CAD: the Contextual Abuse Dataset, NAACL 2021.

**Paper:**  
https://www.aclweb.org/anthology/2021.naacl-main.182/

**Dataset DOI:**  
https://doi.org/10.5281/zenodo.4881008

**Original repository:**  
https://github.com/dongpng/cad_naacl2021

The dataset primarily consists of English Reddit content.

The dataset provides:
- Comment text.
- Primary abuse annotations.
- Secondary annotations.
- Context annotations.
- Target information.
- Conversation and parent-comment information.
- Official training, development and testing splits.

Version 1.1 includes corrections to primary and secondary categories for 95 entries, according to the supplied dataset documentation.

## 2.1 Files found in the CAD archive

The `data.zip` archive contains:

- `cad_v1_1_train.tsv`
- `cad_v1_1_dev.tsv`
- `cad_v1_1_test.tsv`
- `cad_v1_1.tsv`
- `cad_v1_train.tsv`
- `cad_v1_dev.tsv`
- `cad_v1_test.tsv`
- `cad_v1.tsv`
- `cad_codebook_april2021.pdf`
- `README.txt`
- `errata_v1_to_v1_1/`

The project uses CAD v1.1 as the primary dataset version.

CAD v1 is retained for reference and provenance only.

# 3. Dataset Directory Investigation

The raw dataset directory contained multiple datasets and archives.

The following files and datasets were identified:

| Dataset/file | Initial decision |
|---|---|
| CAD v1.1 | Primary dataset |
| CAD v1 | Reference only |
| CADD JSON files | Secondary representation; not selected as primary labels |
| Multi-Sarcasm.csv | External candidate dataset |
| Unhealthy Conversations | External candidate dataset |
| Hindi Hate Speech files | External candidate dataset |
| Parquet shards | Identity and schema require further investigation |
| YouToxic English 1000 | External candidate dataset |
| Abalone dataset | Unrelated; excluded |
| Password-protected HASOC archive | Excluded |
| Other PDF | Not yet investigated |

The original raw datasets are preserved without modification.

Extracted files and their original archives are not treated as separate datasets during analysis.

## 3.1 Password-protected dataset

The `hi_Hasoc2021_train.zip` archive was password-protected.

A decision was made not to use password-protected files.

Therefore, this dataset was excluded rather than attempting to decrypt or extract it.

# 4. Official CAD v1.1 Dataset Structure

The official split files contain three columns:

- `id`
- `text`
- `labels`

The combined metadata file contains 23 columns:

1. `id`
2. `info_id`
3. `info_subreddit`
4. `info_subreddit_id`
5. `info_id.parent`
6. `info_id.link`
7. `info_thread.id`
8. `info_order`
9. `info_image.saved`
10. `annotation_Primary`
11. `annotation_Secondary`
12. `annotation_Context`
13. `annotation_Target`
14. `annotation_Target_top.level.category`
15. `annotation_highlighted`
16. `meta_author`
17. `meta_created_utc`
18. `meta_date`
19. `meta_day`
20. `meta_permalink`
21. `split`
22. `subreddit_seen`
23. `meta_text`

The combined metadata file is particularly useful for investigating conversation relationships and contextual annotations.

# 5. Official Dataset Split Statistics

The official CAD v1.1 split files contain:

| Split | Number of records |
|---|---:|
| Training | 13,584 |
| Development | 4,526 |
| Testing | 5,307 |
| **Total** | **23,417** |

These official split files are used as the source of truth for supervised learning labels.

The combined metadata file contains 27,494 rows and 26,550 unique comment IDs.

The additional metadata rows are not equivalent to additional unique comments.

# 6. Dataset Annotation Categories

The supplied CAD codebook describes six broad primary annotation categories:

1. Neutral
2. Identity-directed abuse
3. Person-directed abuse
4. Counter-speech
5. Non-hateful slurs or reclaimed uses
6. Affiliation-directed abuse

The official v1.1 split files, however, contain five primary classification labels:

| Official label | Description |
|---|---|
| Neutral | Content not classified as abuse |
| IdentityDirectedAbuse | Abuse directed at an identity group |
| AffiliationDirectedAbuse | Abuse directed at an affiliation or associated group |
| PersonDirectedAbuse | Abuse directed at an individual |
| CounterSpeech | Content challenging or condemning abusive language |

**Important distinction:** The six categories described in the codebook should not be confused with the five labels actually present in the official split files.

The `Slur` category does not appear as a separate primary output label in the official v1.1 split files.

The project will therefore initially use the five official labels.

# 7. Official Label Distribution

The official label distribution was calculated independently for training, development and testing.

## 7.1 Training split

| Label | Count | Percentage |
|---|---:|---:|
| Neutral | 11,073 | 81.52% |
| IdentityDirectedAbuse | 1,357 | 9.99% |
| AffiliationDirectedAbuse | 711 | 5.23% |
| PersonDirectedAbuse | 612 | 4.51% |
| CounterSpeech | 113 | 0.83% |
| **Total** | **13,584** | **100% approximately** |

Multi-label records: 265.

## 7.2 Development split

| Label | Count | Percentage |
|---|---:|---:|
| Neutral | 3,692 | 81.57% |
| IdentityDirectedAbuse | 434 | 9.59% |
| AffiliationDirectedAbuse | 246 | 5.44% |
| PersonDirectedAbuse | 205 | 4.53% |
| CounterSpeech | 38 | 0.84% |

Multi-label records: 82.

## 7.3 Testing split

| Label | Count | Percentage |
|---|---:|---:|
| Neutral | 4,342 | 81.82% |
| IdentityDirectedAbuse | 514 | 9.69% |
| AffiliationDirectedAbuse | 242 | 4.56% |
| PersonDirectedAbuse | 237 | 4.47% |
| CounterSpeech | 66 | 1.24% |

Multi-label records: 89.

## 7.4 Key observations

- Neutral is the dominant class, representing approximately 81.5–81.8% of each split.
- Identity-directed abuse is the largest non-neutral category.
- Counter-speech is severely underrepresented.
- The dataset is highly imbalanced.
- Multiple labels can apply to the same comment.

This imbalance means that accuracy alone will not be a sufficient evaluation metric.

The project should investigate:
- Per-class precision.
- Per-class recall.
- Per-class F1-score.
- Macro-F1.
- Micro-F1.
- Confusion matrices for appropriate label-specific analyses.
- Threshold selection.
- Class weighting or other imbalance-handling strategies.

# 8. Multi-label Classification Investigation

The official split files contain comments with more than one primary label.

Examples of observed combinations include:

- IdentityDirectedAbuse + PersonDirectedAbuse
- AffiliationDirectedAbuse + IdentityDirectedAbuse
- AffiliationDirectedAbuse + PersonDirectedAbuse
- CounterSpeech + PersonDirectedAbuse

Training, development and testing contain 265, 82 and 89 multi-label records respectively.

This establishes that ordinary single-class multiclass classification would discard valid annotation information.

### Initial model-design decision

Investigate a five-output multi-label classifier.

Potential architecture:

- Five output units.
- Sigmoid activation.
- Binary cross-entropy-based loss.
- Independent output probabilities.

The Neutral label is exclusive in the observed official combinations. We will preserve the supplied labels and investigate whether a prediction constraint is justified rather than assuming that independent sigmoid outputs will automatically satisfy this convention.

# 9. Duplicate ID Investigation

The combined metadata file contains:

| Measurement | Result |
|---|---:|
| Total metadata rows | 27,494 |
| Unique comment IDs | 26,550 |
| Repeated IDs | 740 |
| Extra duplicate rows | 944 |

Further analysis found:

| Duplicate property | Groups with differences |
|---|---:|
| Text | 0 |
| Primary annotations | 480 |
| Secondary annotations | 591 |
| Context annotations | 139 |
| Split assignment | 0 |
| Parent ID | 0 |
| Thread ID | 0 |
| Conversation order | 0 |

### Finding

Repeated IDs have identical text but may have different annotation values.

For example, the same comment can have different primary annotation values in separate metadata rows.

This means that duplicate rows must not be blindly removed or interpreted as separate independent comments.

### Decision

- Use official split TSVs for supervised learning labels.
- Preserve metadata annotation rows for investigation.
- Use unique comment IDs when counting comments.
- Do not aggregate metadata labels and substitute them for official split labels.

# 10. Official Label Verification

The combined metadata labels were compared against the official train, development and test TSV labels.

All official IDs were found in the corresponding metadata split.

| Split | Official records | Common IDs | Missing IDs | Label mismatches |
|---|---:|---:|---:|---:|
| Train | 13,584 | 13,584 | 0 | 77 |
| Dev | 4,526 | 4,526 | 0 | 31 |
| Test | 5,307 | 5,307 | 0 | 19 |
| **Total** | **23,417** | **23,417** | **0** | **127** |

### Important finding

All 127 observed mismatches had this pattern in the inspected cases:

- Combined metadata primary annotation: `Slur`
- Official split label: `Neutral`

Examples inspected:

- `ehh4rwt`
- `ekynkf8`
- `ehewp6t`
- `eo9lpu4`
- `ei98c97`
- `egspig1`
- `b0sxpa-post`

The inspected comments demonstrate that the presence of a potentially offensive expression or discussion of a slur does not necessarily mean that the entire comment should be classified as abusive.

The exact reason for every disagreement has not yet been established.

### Decision

Official v1.1 split labels will be used as the classification targets.

Metadata annotations will be retained as supplementary research information.

The 127 mismatches should not be silently corrected or overwritten.

# 11. Context Availability Investigation

The combined metadata contains information about parent comments and conversation threads.

Relevant fields:

- `info_id`
- `info_id.parent`
- `info_thread.id`
- `info_order`
- `annotation_Context`
- `meta_text`

Initial context availability audit:

| Measurement | Result |
|---|---:|
| Total metadata rows | 27,494 |
| Unique comment IDs | 26,550 |
| Rows without parent ID | 2,870 |
| Rows with parent available | 14,608 |
| Rows with missing parent | 10,016 |

The presence of a parent ID does not guarantee that the parent comment is available in the metadata.

This creates a significant challenge for context reconstruction.

# 12. Train/Development/Test Context Leakage Investigation

An initial parent-child split audit reported:

| Relationship | Count |
|---|---:|
| Same-split relationships | 13,462 |
| Cross-split relationships | 1,146 |
| Unresolved parent references | 10,016 |

These were the results of the initial relationship audit.

Repeated annotation rows and multiple split assignments at the parent-ID level must be handled carefully when interpreting these counts.

### Why this matters

Suppose a training comment has a parent comment in the test split.

If the model uses test information while training, this may introduce data leakage and lead to an unreliable evaluation.

### Decision

Conversation context reconstruction must respect official split boundaries.

A context-aware model must not use held-out test labels or information in a way that leaks evaluation targets into training.

A more detailed leakage-safe reconstruction audit is required before implementing context features.

# 13. Conversation and Thread Investigation

The metadata provides:

- Parent-child relationships through `info_id.parent`.
- Thread grouping through `info_thread.id`.
- Conversation ordering through `info_order`.

Sampled threads confirmed that replies can be linked to earlier comments.

For example, a conversation can contain:

1. A root post.
2. A reply containing person-directed abuse.
3. A neutral response.
4. Another reply containing person-directed abuse.

This demonstrates that comment meaning and annotation can depend on preceding conversation content.

However, the full conversation history is not available for every comment.

Therefore, the model must distinguish between:
- Available context.
- Missing parent context.
- Root comments.
- Replies with unavailable ancestors.

# 14. Context Annotation Investigation

The `annotation_Context` field contains values such as:

- `CurrentContent`
- `PreviousContent`
- `NA`

Initial metadata row counts:

| Context annotation | Rows |
|---|---:|
| NA | 21,935 |
| CurrentContent | 3,993 |
| PreviousContent | 1,566 |

After grouping by unique comment IDs, the observed context combinations were:

| Context combination | Unique comments |
|---|---:|
| No context annotation | 21,935 |
| CurrentContent only | 3,117 |
| PreviousContent only | 1,359 |
| Both | 139 |

### Important distinction

The context annotation does not directly provide a separate context-text field.

It indicates whether current or previous content was relevant to the annotation.

It must not be interpreted as proof that the complete conversation history is available.

# 15. Difficulties Encountered

The following challenges were identified during dataset investigation.

## 15.1 Multiple dataset representations

The raw directory contains multiple formats and versions of the dataset, including JSON, TSV and ZIP archives.

This created a risk of using inconsistent representations.

**Resolution:** Use official CAD v1.1 TSV split files as the primary source of training labels.

## 15.2 Duplicate comment IDs

The combined metadata contains repeated IDs with different annotation values.

**Resolution:** Preserve unique comment identity and use official split labels for supervised learning.

## 15.3 Label disagreement

There are 127 observed mismatches between aggregated metadata annotations and official split labels.

**Resolution:** Use the official split labels as the supervised targets and retain the discrepancies for further investigation.

## 15.4 Severe class imbalance

Neutral dominates the dataset, while CounterSpeech has very few examples.

**Resolution:** Plan class-wise evaluation and investigate imbalance-handling techniques.

## 15.5 Missing parent comments

Many comments reference parent IDs that are not present in the combined metadata.

**Resolution:** Missing context must be represented explicitly rather than fabricated.

## 15.6 Cross-split conversation relationships

Some parent-child relationships cross train/dev/test boundaries.

**Resolution:** A leakage-safe context reconstruction strategy is required.

## 15.7 Incomplete conversation histories

The dataset provides conversation identifiers and parent references, but complete histories cannot be reconstructed for every record.

**Resolution:** Context availability must be measured and considered in model evaluation.

## 15.8 Password-protected archive

The Hindi HASOC archive was password-protected.

**Resolution:** Excluded from the project.

# 16. Current Research Decisions

| Area | Current decision |
|---|---|
| Primary dataset | CAD v1.1 |
| Training labels | Official split TSV labels |
| Number of output labels | Five |
| Classification formulation | Multi-label |
| Duplicate handling | Preserve annotation records; do not blindly deduplicate |
| Metadata usage | Context and conversation information |
| CAD v1 | Reference only |
| Raw data | Preserve unchanged |
| Evaluation splits | Official train/dev/test |
| Context reconstruction | Pending leakage-safe design |
| Imbalance handling | To be investigated |
| Model training | Not started |

# 17. Audit Scripts

All dataset investigation functionality is consolidated into:

`src/audit_dataset.py`

The script includes:

1. Dataset inventory.
2. Archive inspection.
3. Official TSV structure inspection.
4. Duplicate ID analysis.
5. Metadata label distribution.
6. Context availability.
7. Conversation/thread inspection.
8. Context annotation inspection.
9. Official label verification.
10. Official label distribution.
11. Final audit summary.

Run using:

```bash
python src/audit_dataset.py
```

The audit script does not modify the original raw datasets.

# 18. Work Completed So Far

- Identified the primary dataset and its source.
- Inspected the CAD archive.
- Investigated the official v1.1 split files.
- Inspected the combined metadata structure.
- Verified all official split IDs against metadata.
- Investigated repeated comment IDs.
- Identified metadata annotation disagreements.
- Calculated official label distributions.
- Confirmed the presence of multi-label records.
- Investigated context annotation values.
- Inspected conversation and parent-child relationships.
- Identified missing parent references.
- Identified potential cross-split context leakage.
- Consolidated the audit functionality into one script.

# 19. Pending Work

The following tasks remain:

1. Verify the consolidated audit script against previous audit outputs.
2. Design leakage-safe context reconstruction.
3. Investigate the 127 metadata/official-label mismatches further.
4. Finalize the target representation and Neutral-label prediction convention.
5. Investigate class imbalance handling.
6. Develop the preprocessing pipeline.
7. Build context-aware and context-free baseline models.
8. Develop the proposed context-aware architecture.
9. Perform comparative experiments.
10. Evaluate per-class performance and contextual robustness.
11. Perform ablation studies.
12. Document limitations and research contributions.

# 20. Current Project Status

**Dataset investigation:** Initial audit completed.

**Label source:** Official CAD v1.1 split TSVs.

**Context investigation:** Initial analysis completed; reconstruction strategy pending.

**Preprocessing:** Not started.

**Model development:** Not started.

**Evaluation:** Not started.

The next research milestone is to develop a leakage-safe method for reconstructing conversational context before building the preprocessing and model pipelines.

# 21. Leakage-Safe Context Reconstruction Audit

## 21.1 Objective

The objective of this investigation was to determine how much conversational context can be reconstructed from the CAD v1.1 metadata while respecting the official training, development and testing boundaries.

The investigation used:

- `info_id`: Comment identifier.
- `info_id.parent`: Parent comment identifier.
- `info_thread.id`: Conversation thread identifier.
- `info_order`: Position in the conversation.
- `meta_text`: Comment text.
- `split`: Dataset split assignment.

The official train/dev/test TSV files were used to establish the supervised dataset boundaries.

## 21.2 Metadata consistency

The reconstruction audit confirmed:

| Measurement | Result |
|---|---:|
| Unique metadata IDs | 26,550 |
| Repeated IDs | 740 |
| IDs with conflicting parent IDs | 0 |
| IDs with conflicting text | 0 |
| IDs with conflicting split assignments | 0 |

Repeated annotation rows therefore did not introduce conflicting parent IDs, text or split assignments in this audit.

## 21.3 Context availability results

The audit covered all 23,417 official records.

| Context status | Records |
|---|---:|
| Same-split parent text available | 12,835 |
| Parent missing from metadata | 8,522 |
| No parent ID | 1,660 |
| Parent excluded from official splits | 396 |
| Parent text missing | 4 |
| **Total** | **23,417** |

Approximately 54.81% of the official records have usable parent text under the initial strict same-split policy.

Approximately 45.19% do not have usable parent text under this policy.

## 21.4 Split-wise context availability

| Measurement | Train | Dev | Test |
|---|---:|---:|---:|
| Official records | 13,584 | 4,526 | 5,307 |
| No parent ID | 880 | 357 | 423 |
| Parent missing from metadata | 4,880 | 1,712 | 1,930 |
| Same-split parent available | 7,570 | 2,390 | 2,875 |
| Parent excluded from official splits | 251 | 66 | 79 |
| Parent text missing | 3 | 1 | 0 |

The training, development and testing sets all contain records with reconstructable parent context.

## 21.5 Cross-split relationship analysis

The audit examined official parent-child relationships.

Results:

- Unique official parent-child relationships: 12,836.
- Cross-split relationships: 0.

No parent-child relationship crossing the official train/dev/test boundaries was identified in this audit.

This is an encouraging result for leakage-safe context reconstruction.

Nevertheless, the implementation must explicitly verify split assignments before using any parent text.

## 21.6 Missing context investigation

The largest limitation is the absence of parent comments.

A total of 8,522 official records reference parents that are not available in the combined metadata.

Additionally:

- 1,660 records have no parent ID.
- 396 records have parents present in metadata but excluded from the official splits.
- 4 records have parent records with missing text.

The project will not fabricate missing parent text.

## 21.7 Context annotation versus context availability

The `annotation_Context` field was also examined across official records.

| Annotation | Train | Dev | Test |
|---|---:|---:|---:|
| No context annotation | 11,010 | 3,664 | 4,325 |
| CurrentContent | 1,862 | 564 | 643 |
| PreviousContent | 804 | 318 | 362 |

The context annotation values are counted separately. A comment may have more than one annotation, so these counts are not necessarily mutually exclusive.

An important distinction is that annotation context and reconstructable parent text are different concepts.

A comment with no context annotation may still have a usable parent comment. Conversely, a comment annotated as context-dependent may have an unavailable parent.

## 21.8 Research interpretation

The investigation establishes that CAD v1.1 supports partial conversation reconstruction.

Approximately 55% of the official records can be paired with same-split parent text using the initial strict policy.

This suggests that a context-aware model is feasible, but it must handle missing context explicitly.

Discarding all records without parent text would remove a substantial portion of the dataset and could introduce selection bias.

Therefore, the project will investigate a context-aware approach that can operate with and without available parent context.

## 21.9 Initial context-handling decisions

1. Use official train/dev/test labels without modification.
2. Use parent text only when the parent belongs to the same official split.
3. Do not use parent labels as model input.
4. Do not fabricate missing parent text.
5. Preserve records without context.
6. Maintain a context-availability indicator.
7. Investigate excluded parents separately.
8. Evaluate performance separately on context-available and context-unavailable records.

## 21.10 Important limitation

The current audit establishes parent-text availability, not complete conversation-history availability.

A single parent comment may not contain all the information required to interpret a target comment.

Further investigation is required to determine whether multi-level conversation history can be reconstructed reliably and whether it adds useful information beyond the immediate parent.

## 21.11 Next milestone

The next milestone is to investigate parent-chain reconstruction, context completeness and the relationship between human context annotations and actual parent-text availability.

Only after this investigation will the project finalize the context input representation.

## 22. Corrected Conversation Depth and History Audit

### 22.1 Objective

The purpose of this audit was to investigate how much conversational history can be reconstructed for each official CAD v1.1 target comment. The analysis examines conversation depth, parent availability, context annotations and thread consistency.

### 22.2 Dataset Counting Validation

The corrected script counts each unique official comment ID exactly once, using the official train, development and test files as the source of truth.

| Metric | Count |
|---|---:|
| Raw metadata rows | 27,494 |
| Unique metadata IDs | 26,550 |
| Unique official records | 23,417 |
| Train records | 13,584 |
| Development records | 4,526 |
| Test records | 5,307 |

All 23,417 official IDs were found in the metadata. The official record count validation passed.

### 22.3 Conversation Depth Distribution

Conversation depth represents the number of consecutive ancestors that can be reconstructed using available metadata, non-empty parent text and the same official data split.

| Available ancestors | Records | Percentage |
|---|---:|---:|
| 0 | 10,582 | 45.19% |
| 1 | 5,830 | 24.90% |
| 2 | 3,817 | 16.30% |
| 3 | 1,733 | 7.40% |
| 4 | 804 | 3.43% |
| 5 | 513 | 2.19% |
| 6 or more | 138 | 0.59% |
| **Total** | **23,417** | **100%** |

Key findings:

- 12,835 records (54.81%) have at least one reconstructable ancestor.
- 7,005 records (29.91%) have at least two reconstructable ancestors.
- 3,188 records (13.62%) have at least three reconstructable ancestors.
- The maximum observed reconstructable depth was 13 ancestors.
- Average reconstructable ancestors per official record: 1.0898.

### 22.4 Conversation Termination Reasons

| Termination reason | Count | Percentage |
|---|---:|---:|
| Parent missing from metadata | 21,032 | 89.82% |
| Root or no parent | 1,660 | 7.09% |
| Parent excluded from official splits | 720 | 3.07% |
| Parent text missing | 5 | 0.02% |
| **Total** | **23,417** | **100%** |

The termination reason describes why reconstruction stopped. It does not necessarily indicate that no ancestors were available. A chain may contain one or more ancestors before reaching a missing parent.

### 22.5 Context Annotation versus Actual Parent Availability

| Context annotation | Parent available | Parent unavailable |
|---|---:|---:|
| No context annotation | 10,357 | 8,642 |
| CurrentContent | 1,569 | 1,365 |
| PreviousContent | 825 | 524 |
| CurrentContent + PreviousContent | 84 | 51 |

Total unique official records compared: 23,417.

This demonstrates that context annotations and reconstructable parent text are separate properties. Some records have no context annotation but still have available parent text, while others have annotations but no reconstructable parent text.

The proposed dataset representation should preserve these properties separately.

### 22.6 Thread Consistency

| Metric | Result |
|---|---:|
| Unique threads | 1,395 |
| Parent-child relationships checked | 14,062 |
| Relationships with missing thread IDs | 0 |
| Parent-child thread ID mismatches | 0 |

No thread ID mismatches were detected among the checked metadata relationships.

### 22.7 Research Implications

The audit establishes that conversational context is available for a substantial portion of the dataset, but complete conversation history is not available for every record.

The following configurations are proposed for later experimental comparison:

1. Current comment only.
2. Current comment plus immediate parent.
3. Current comment plus up to two ancestors.
4. Current comment plus up to three ancestors.

The experiments will determine whether additional conversational history improves classification.

### 22.8 Decisions and Safeguards

1. Count every official comment ID exactly once.
2. Use official train/dev/test files as the source of truth.
3. Use only same-split ancestor text.
4. Do not use ancestor labels as model inputs.
5. Do not fabricate missing conversation history.
6. Preserve comments with unavailable context.
7. Keep context annotations separate from actual parent availability.
8. Preserve official target labels.
9. Evaluate context-available and context-unavailable records separately.
10. Compare context configurations experimentally rather than assuming longer history is better.

### 22.9 Audit Status

The corrected conversation-depth audit completed successfully. All official record counts were validated, and the depth distribution and annotation comparison each accounted for exactly 23,417 unique official records.