
# ============================================================
# CONTEXT-AWARE TOXICITY CLASSIFIER
# STEP 2.1: TEXT QUALITY INSPECTION
#
# Dataset: Contextual Abuse Dataset (CAD) v1.1
#
# Purpose:
# Inspect text quality before applying preprocessing.
# Original dataset files are never modified.
# ============================================================

import re
import zipfile
import html
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

ARCHIVE_PATH = RAW_DIR / "data.zip"

SPLIT_FILES = {
    "train": "cad_v1_1_train.tsv",
    "dev": "cad_v1_1_dev.tsv",
    "test": "cad_v1_1_test.tsv"
}

EXPECTED_LABELS = [
    "Neutral",
    "IdentityDirectedAbuse",
    "AffiliationDirectedAbuse",
    "PersonDirectedAbuse",
    "CounterSpeech"
]

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. REGULAR EXPRESSIONS
# ============================================================

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE
)

USERNAME_PATTERN = re.compile(
    r"(?<!\w)u/[A-Za-z0-9_-]+|"
    r"(?<!\w)@[A-Za-z0-9_-]+",
    re.IGNORECASE
)

SUBREDDIT_PATTERN = re.compile(
    r"(?<!\w)r/[A-Za-z0-9_-]+",
    re.IGNORECASE
)

HTML_ENTITY_PATTERN = re.compile(
    r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);"
)

HTML_TAG_PATTERN = re.compile(
    r"<[^>]+>"
)

MARKDOWN_PATTERN = re.compile(
    r"[*_~`]{1,3}"
)

REDDIT_QUOTE_PATTERN = re.compile(
    r"(?m)^\s*&gt;|^\s*>"
)

WHITESPACE_PATTERN = re.compile(r"\s+")


# ============================================================
# 3. LOAD OFFICIAL CAD SPLITS FROM ZIP
# ============================================================

def load_dataset():

    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Dataset archive not found: {ARCHIVE_PATH}"
        )

    data = {}

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:

        archive_names = archive.namelist()

        for split, filename in SPLIT_FILES.items():

            matches = [
                name for name in archive_names
                if Path(name).name == filename
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one {filename} in data.zip. "
                    f"Found: {matches}"
                )

            with archive.open(matches[0]) as file:

                df = pd.read_csv(
                    file,
                    sep="\t",
                    dtype=str,
                    keep_default_na=False,
                    encoding="utf-8-sig"
                )

            required_columns = {"id", "text", "labels"}

            if not required_columns.issubset(df.columns):
                raise ValueError(
                    f"Unexpected columns in {filename}: "
                    f"{list(df.columns)}"
                )

            df["split"] = split
            data[split] = df

            print(
                f"Loaded {split.upper():5s}: "
                f"{len(df):,} records"
            )

    return data


# ============================================================
# 4. BASIC TEXT QUALITY STATISTICS
# ============================================================

def inspect_basic_quality(data):

    results = []

    for split, df in data.items():

        text = df["text"].fillna("")

        lengths = text.str.len()

        results.append({

            "split": split,

            "total_records": len(df),

            "missing_text": int(df["text"].isna().sum()),

            "empty_text": int(text.str.strip().eq("").sum()),

            "duplicate_text_rows": int(
                text[text.duplicated(keep=False)].shape[0]
            ),

            "unique_texts": int(text.nunique()),

            "minimum_length": int(lengths.min()),

            "maximum_length": int(lengths.max()),

            "mean_length": round(lengths.mean(), 2),

            "median_length": round(lengths.median(), 2),

            "std_length": round(lengths.std(), 2),

            "texts_below_10_chars": int(
                (lengths < 10).sum()
            ),

            "texts_below_20_chars": int(
                (lengths < 20).sum()
            ),

            "texts_above_500_chars": int(
                (lengths > 500).sum()
            ),

            "texts_above_1000_chars": int(
                (lengths > 1000).sum()
            )

        })

    result = pd.DataFrame(results)

    result.to_csv(
        REPORT_DIR / "text_quality_summary.csv",
        index=False
    )

    print("\n========== BASIC TEXT QUALITY ==========")
    print(result.to_string(index=False))

    return result


# ============================================================
# 5. INSPECT SPECIAL TEXT PATTERNS
# ============================================================

def inspect_text_patterns(data):

    results = []

    for split, df in data.items():

        texts = df["text"].fillna("")

        results.append({

            "split": split,

            "texts_with_urls": int(
                texts.str.contains(URL_PATTERN).sum()
            ),

            "texts_with_usernames": int(
                texts.str.contains(USERNAME_PATTERN).sum()
            ),

            "texts_with_subreddit_mentions": int(
                texts.str.contains(SUBREDDIT_PATTERN).sum()
            ),

            "texts_with_html_entities": int(
                texts.str.contains(HTML_ENTITY_PATTERN).sum()
            ),

            "texts_with_html_tags": int(
                texts.str.contains(HTML_TAG_PATTERN).sum()
            ),

            "texts_with_markdown_symbols": int(
                texts.str.contains(MARKDOWN_PATTERN).sum()
            ),

            "texts_with_quote_formatting": int(
                texts.str.contains(REDDIT_QUOTE_PATTERN).sum()
            ),

            "texts_with_newlines": int(
                texts.str.contains(r"\n").sum()
            ),

            "texts_with_tabs": int(
                texts.str.contains(r"\t").sum()
            ),

            "texts_with_non_ascii": int(
                texts.apply(
                    lambda x: any(ord(c) > 127 for c in x)
                ).sum()
            ),

            "texts_with_repeated_punctuation": int(
                texts.str.contains(r"([!?.,])\1{1,}").sum()
            )

        })

    result = pd.DataFrame(results)

    result.to_csv(
        REPORT_DIR / "text_pattern_summary.csv",
        index=False
    )

    print("\n========== TEXT PATTERN INSPECTION ==========")
    print(result.to_string(index=False))

    return result


# ============================================================
# 6. LABEL-WISE TEXT LENGTH ANALYSIS
# ============================================================

def inspect_label_lengths(data):

    all_data = pd.concat(
        data.values(),
        ignore_index=True
    )

    results = []

    for label in EXPECTED_LABELS:

        # Preserve multi-label records.
        # A record contributes to every label it contains.
        mask = all_data["labels"].apply(
            lambda value: label in [
                item.strip()
                for item in value.split(",")
            ]
        )

        subset = all_data.loc[mask].copy()

        lengths = subset["text"].str.len()

        results.append({

            "label": label,

            "records": len(subset),

            "mean_length": round(lengths.mean(), 2),

            "median_length": round(lengths.median(), 2),

            "minimum_length": int(lengths.min()),

            "maximum_length": int(lengths.max()),

            "texts_below_20_chars": int(
                (lengths < 20).sum()
            ),

            "texts_above_500_chars": int(
                (lengths > 500).sum()
            )

        })

    result = pd.DataFrame(results)

    result.to_csv(
        REPORT_DIR / "label_text_length_summary.csv",
        index=False
    )

    print("\n========== LABEL-WISE TEXT LENGTH ==========")
    print(result.to_string(index=False))

    return result


# ============================================================
# 7. CROSS-SPLIT DUPLICATE INSPECTION
# ============================================================

def inspect_cross_split_duplicates(data):

    all_data = pd.concat(
        data.values(),
        ignore_index=True
    )

    text_groups = (
        all_data.groupby("text")["split"]
        .agg(lambda values: sorted(set(values)))
        .reset_index()
    )

    cross_split = text_groups[
        text_groups["split"].apply(len) > 1
    ].copy()

    cross_split["splits"] = cross_split["split"].apply(
        lambda values: ", ".join(values)
    )

    cross_split = cross_split[
        ["text", "splits"]
    ]

    cross_split.to_csv(
        REPORT_DIR / "cross_split_duplicate_texts.csv",
        index=False
    )

    print("\n========== CROSS-SPLIT DUPLICATES ==========")
    print(
        f"Unique comment texts appearing in multiple splits: "
        f"{len(cross_split):,}"
    )

    if not cross_split.empty:

        print("\nSample cross-split duplicates:")

        for _, row in cross_split.head(5).iterrows():

            print("\nSplits:", row["splits"])
            print("Text:", row["text"][:250])

    return cross_split


# ============================================================
# 8. REPRESENTATIVE EXAMPLES BY LABEL
# ============================================================

def save_label_examples(data):

    all_data = pd.concat(
        data.values(),
        ignore_index=True
    )

    examples = []

    for label in EXPECTED_LABELS:

        subset = all_data[
            all_data["labels"].apply(
                lambda value: label in value.split("|")
            )
        ]

        # Deterministic sampling.
        sample = subset.head(10)

        for _, row in sample.iterrows():

            examples.append({

                "label": label,

                "split": row["split"],

                "id": row["id"],

                "text": row["text"],

                "labels": row["labels"]

            })

    result = pd.DataFrame(examples)

    result.to_csv(
        REPORT_DIR / "representative_label_examples.csv",
        index=False
    )

    print("\n========== REPRESENTATIVE EXAMPLES ==========")

    for label in EXPECTED_LABELS:

        print(f"\n--- {label} ---")

        subset = result[result["label"] == label]

        for _, row in subset.head(3).iterrows():

            print("-", row["text"][:250])


# ============================================================
# 9. TEXT LENGTH DISTRIBUTION BY SPLIT
# ============================================================

def save_length_distribution(data):

    all_data = pd.concat(
        data.values(),
        ignore_index=True
    )

    result = all_data[
        ["id", "split", "labels", "text"]
    ].copy()

    result["text_length"] = result["text"].str.len()

    result.to_csv(
        REPORT_DIR / "text_length_distribution.csv",
        index=False
    )

    print("\nSaved record-level text lengths.")


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("CAD v1.1 - TEXT QUALITY INSPECTION")
    print("=" * 65)

    print("\nSTEP 1: Loading official dataset splits")

    data = load_dataset()

    print("\nSTEP 2: Basic text quality inspection")

    inspect_basic_quality(data)

    print("\nSTEP 3: Inspecting special text patterns")

    inspect_text_patterns(data)

    print("\nSTEP 4: Label-wise text length analysis")

    inspect_label_lengths(data)

    print("\nSTEP 5: Cross-split duplicate inspection")

    inspect_cross_split_duplicates(data)

    print("\nSTEP 6: Saving representative label examples")

    save_label_examples(data)

    print("\nSTEP 7: Saving text-length distribution")

    save_length_distribution(data)

    print("\n" + "=" * 65)
    print("TEXT QUALITY INSPECTION COMPLETED")
    print("=" * 65)

    print("\nReports saved in:")
    print(REPORT_DIR)


if __name__ == "__main__":
    main()
