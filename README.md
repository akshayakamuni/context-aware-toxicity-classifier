# Context-Aware Toxicity Classifier

## 1. Overview

The **Context-Aware Toxicity Classifier** is an NLP project that investigates whether conversational context improves the classification of abusive and toxic online comments.

A comment may appear toxic when viewed independently but have a different meaning when its surrounding conversation is considered. For example, a comment might quote offensive language to oppose it rather than express abuse.

This project compares traditional machine learning, transformer-based text classification, and context-aware transformer architectures.

## 2. Objectives

- Compare comment-only and context-aware toxicity classification.
- Investigate the role of conversational context in identifying abusive language and counter-speech.
- Evaluate multiple machine learning and deep learning approaches.
- Analyze performance across different toxicity categories.
- Examine the limitations of text-only and context-aware classification.

## 3. Dataset

**Dataset:** Contextual Abuse Dataset (CAD)

The dataset contains Reddit comments with five multi-label categories:

| Label | Description |
|---|---|
| Neutral | Non-abusive content |
| IdentityDirectedAbuse | Abuse directed towards an identity |
| AffiliationDirectedAbuse | Abuse directed towards an affiliation |
| PersonDirectedAbuse | Abuse directed towards an individual |
| CounterSpeech | Content opposing or challenging abusive language |

The classification task uses multi-label predictions, meaning a comment can belong to multiple categories.

### Dataset Splits

| Split | Original dataset |
|---|---:|
| Training | 13,584 |
| Development | 4,526 |
| Testing | 5,307 |
| Total | 23,417 |

For the context-aware experiment, parent-child comment pairs were constructed within the original dataset splits.

| Split | Context-paired examples |
|---|---:|
| Training | 7,571 |
| Development | 2,390 |
| Testing | 2,875 |

## 4. Model Architectures and Experiments

### E1: TF-IDF + Logistic Regression

A traditional machine learning baseline.

- Text representation: TF-IDF.
- Unigrams and bigrams.
- One-vs-Rest Logistic Regression.
- Class-weight balancing.
- Multi-label classification.

### E2: BERT Text-Only Classification

A transformer-based baseline using only the current comment.

- Pretrained model: `bert-base-uncased`.
- Maximum sequence length: 256.
- Classification representation: CLS token.
- Dropout and fully connected classification layer.
- Five-output multi-label prediction.
- Loss function: Binary Cross-Entropy with Logits.

### E3: BERT Context-Aware Classification

A context-aware architecture that processes the current comment and its parent comment using a shared BERT encoder.

The two representations are combined using:

- Current comment representation.
- Parent comment representation.
- Absolute difference between representations.
- Element-wise product of representations.

The resulting fused representation is passed through a feed-forward classification network to predict the five labels.

**Fusion equation:**

h = [h_current ; h_parent ; |h_current - h_parent| ; h_current ⊙ h_parent]

Where:
- h_current represents the current comment.
- h_parent represents the parent comment.
- ⊙ denotes element-wise multiplication.
- ; denotes concatenation.

## 5. Experimental Results

E2 and E3 are compared on the same paired test subset to ensure a fair comparison.

| Metric | E2: BERT Text-Only | E3: Context-Aware BERT |
|---|---:|---:|
| Macro-F1 | 0.4413 | 0.4463 |
| Micro-F1 | 0.8242 | 0.8175 |
| Macro Precision | 0.4280 | 0.4412 |
| Macro Recall | 0.4578 | 0.4545 |
| Hamming Loss | 0.0751 | 0.0779 |
| Subset Accuracy | 0.7770 | 0.7777 |

### Observations

- E3 achieves a higher Macro-F1 score than E2.
- E3 improves macro precision.
- E2 achieves higher Micro-F1 and macro recall.
- E3 improves the F1 scores for AffiliationDirectedAbuse and CounterSpeech.
- Context integration does not improve every label or every evaluation metric.

These results demonstrate that conversational context can affect classification performance, but its benefits are not uniform across categories.

## 6. Repository Structure

```text
context-aware-toxicity-classifier/
│
├── configs/
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── docs/
├── models/
│
├── notebooks/
│   └── E3_context_aware_BERT.ipynb
│
├── reports/
│   ├── e1_tfidf_logreg/
│   ├── e2_bert_text_only/
│   └── e3_context_aware/
│
├── src/
│   ├── models/
│   ├── preprocessing/
│   ├── data/
│   └── evaluation/
│
├── tests/
├── README.md
├── requirements.txt
├── check_environment.py
└── .gitignore
```

## 7. Model Checkpoints

Trained model checkpoints and associated evaluation artifacts are available through GitHub Releases.

**[Download Model Checkpoints – GitHub Releases](https://github.com/akshayakamuni/context-aware-toxicity-classifier/releases)**

The release contains the uploaded model checkpoints and supporting artifacts.

The E3 model requires the corresponding model architecture and preprocessing implementation from the repository.

## 8. Installation

Clone the repository:

```bash
git clone https://github.com/akshayakamuni/context-aware-toxicity-classifier.git

cd context-aware-toxicity-classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 9. Reproducibility

The repository includes:

- E1 and E2 Python training implementations.
- E3 Google Colab training notebook.
- Dataset preprocessing and auditing scripts.
- Evaluation reports and prediction artifacts.
- Model checkpoints through GitHub Releases.

The CAD dataset should be obtained and stored according to its applicable licensing and distribution requirements.

## 10. Project Status

| Component | Status |
|---|---|
| Dataset preprocessing | Completed |
| E1: TF-IDF + Logistic Regression | Completed |
| E2: BERT Text-Only | Completed |
| E3: BERT Context-Aware | Completed |
| Model evaluation | Completed |
| E2 vs E3 comparison | Completed |
| Model checkpoints | Published in GitHub Releases |
| Project documentation | In progress |

## 11. Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy
- Google Colab
- Visual Studio Code
- Git and GitHub
