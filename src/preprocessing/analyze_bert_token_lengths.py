"""
CAD v1.1 - BERT Token-Length Analysis

Purpose:
1. Measure BERT token lengths of current comments.
2. Reconstruct immediate parent comments from CAD metadata.
3. Measure parent-comment token lengths.
4. Measure combined current-parent pair lengths.
5. Estimate truncation at 64, 128, 256 and 512 tokens.
6. Produce overall, split-wise and label-wise reports.

Important:
- Uses official CAD v1.1 train/dev/test labels.
- Uses only same-split parent comments.
- Does not modify raw data.
- Does not truncate or remove any records.
"""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from transformers import AutoTokenizer


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ARCHIVE = (
    PROJECT_ROOT / "datasets" / "raw" / "data.zip"
)

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TOKENIZER_NAME = "bert-base-uncased"

MAX_LENGTHS = [64, 128, 256, 512]

BATCH_SIZE = 256


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def read_tsv_from_zip(archive, member):
    """Read a TSV file directly from the dataset ZIP."""

    with archive.open(member) as file:

        return pd.read_csv(
            file,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8"
        )


def find_metadata_member(archive):
    """Locate the CAD v1.1 metadata file."""

    candidates = [
        name
        for name in archive.namelist()
        if Path(name).name.lower() == "cad_v1_1.tsv"
    ]

    if not candidates:
        raise FileNotFoundError(
            "cad_v1_1.tsv not found inside data.zip"
        )

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one metadata file, found: {candidates}"
        )

    return candidates[0]


def find_split_member(archive, split):
    """Locate one official split TSV."""

    candidates = []

    for name in archive.namelist():

        if not name.lower().endswith(".tsv"):
            continue

        basename = Path(name).name.lower()

        if split not in basename:
            continue

        if "v1_1" not in basename and "v1.1" not in basename:
            continue

        try:

            with archive.open(name) as file:

                columns = pd.read_csv(
                    file,
                    sep="\t",
                    nrows=0
                ).columns.tolist()

            if {"id", "text", "labels"}.issubset(
                set(columns)
            ):
                candidates.append(name)

        except Exception:
            continue

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one official {split} file. "
            f"Found: {candidates}"
        )

    return candidates[0]


def clean_text(value):
    """Convert missing metadata values to an empty string."""

    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in {
        "",
        "na",
        "nan",
        "none",
        "null"
    }:
        return ""

    return value


def tokenize_lengths(tokenizer, texts, batch_size):
    """
    Return token counts for a list of texts.

    Special tokens are included in the count.
    No truncation or padding is applied.
    """

    lengths = []

    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False
        )

        batch_lengths = [
            len(input_ids)
            for input_ids in encoded["input_ids"]
        ]

        lengths.extend(batch_lengths)

    return lengths


def tokenize_pair_lengths(
    tokenizer,
    current_texts,
    parent_texts,
    batch_size
):
    """
    Measure token lengths of current-parent pairs.

    BERT pair special tokens are included.
    No truncation or padding is applied.
    """

    lengths = []

    for start in range(0, len(current_texts), batch_size):

        current_batch = current_texts[
            start:start + batch_size
        ]

        parent_batch = parent_texts[
            start:start + batch_size
        ]

        encoded = tokenizer(
            current_batch,
            text_pair=parent_batch,
            add_special_tokens=True,
            truncation=False,
            padding=False
        )

        batch_lengths = [
            len(input_ids)
            for input_ids in encoded["input_ids"]
        ]

        lengths.extend(batch_lengths)

    return lengths


def summarize_lengths(series):
    """Calculate useful descriptive statistics."""

    values = series.dropna()

    if len(values) == 0:

        return {
            "records": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "maximum": None
        }

    return {
        "records": len(values),
        "mean": round(values.mean(), 2),
        "median": round(values.median(), 2),
        "p90": round(values.quantile(0.90), 2),
        "p95": round(values.quantile(0.95), 2),
        "p99": round(values.quantile(0.99), 2),
        "maximum": int(values.max())
    }


def truncation_count(series, max_length):
    """Count records that exceed a given token limit."""

    values = series.dropna()

    if len(values) == 0:
        return 0

    return int((values > max_length).sum())


# ============================================================
# 3. LOAD TOKENIZER
# ============================================================

print("=" * 65)
print("CAD v1.1 - BERT TOKEN-LENGTH ANALYSIS")
print("=" * 65)

print("\nLoading tokenizer:", TOKENIZER_NAME)

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    token=False
)

print("Tokenizer loaded successfully.")


# ============================================================
# 4. LOAD OFFICIAL DATA AND METADATA
# ============================================================

if not RAW_ARCHIVE.exists():

    raise FileNotFoundError(
        f"Dataset archive not found: {RAW_ARCHIVE}"
    )

with ZipFile(RAW_ARCHIVE, "r") as archive:

    metadata_member = find_metadata_member(archive)

    metadata = read_tsv_from_zip(
        archive,
        metadata_member
    )

    official_frames = []

    for split in ["train", "dev", "test"]:

        member = find_split_member(
            archive,
            split
        )

        frame = read_tsv_from_zip(
            archive,
            member
        )

        frame["official_split"] = split

        official_frames.append(frame)


official = pd.concat(
    official_frames,
    ignore_index=True
)

official["id"] = official["id"].map(clean_text)
official["text"] = official["text"].map(clean_text)
official["labels"] = official["labels"].map(clean_text)

print(
    "\nOfficial records loaded:",
    len(official)
)


# ============================================================
# 5. PREPARE METADATA LOOKUP
# ============================================================

required_columns = {
    "info_id",
    "info_id.parent",
    "meta_text"
}

missing = required_columns - set(metadata.columns)

if missing:

    raise ValueError(
        f"Metadata is missing columns: {missing}"
    )

metadata["info_id"] = (
    metadata["info_id"].map(clean_text)
)

metadata["info_id.parent"] = (
    metadata["info_id.parent"].map(clean_text)
)

metadata["meta_text"] = (
    metadata["meta_text"].map(clean_text)
)

# Earlier audits established that parent IDs and comment text
# are consistent across repeated metadata rows.
# Context annotations are not used in this script.

metadata_unique = (
    metadata.drop_duplicates(
        subset=["info_id"],
        keep="first"
    )
    .set_index("info_id")
)

# Map each official comment ID to its official split.
official_split_lookup = (
    official.set_index("id")["official_split"].to_dict()
)


# ============================================================
# 6. RECONSTRUCT SAME-SPLIT PARENT TEXT
# ============================================================

parent_texts = []
parent_statuses = []

for _, row in official.iterrows():

    comment_id = row["id"]
    current_split = row["official_split"]

    parent_text = ""
    parent_status = ""

    if comment_id not in metadata_unique.index:

        parent_status = "current_metadata_missing"

    else:

        parent_id = clean_text(
            metadata_unique.loc[
                comment_id,
                "info_id.parent"
            ]
        )

        if not parent_id:

            parent_status = "no_parent_id"

        elif parent_id not in metadata_unique.index:

            parent_status = "parent_missing_from_metadata"

        else:

            candidate_parent_text = clean_text(
                metadata_unique.loc[
                    parent_id,
                    "meta_text"
                ]
            )

            parent_official_split = (
                official_split_lookup.get(
                    parent_id,
                    ""
                )
            )

            if not candidate_parent_text:

                parent_status = "parent_text_missing"

            elif parent_official_split != current_split:

                if parent_official_split:

                    parent_status = (
                        "parent_in_different_official_split"
                    )

                else:

                    parent_status = (
                        "parent_not_in_official_splits"
                    )

            else:

                parent_text = candidate_parent_text

                parent_status = (
                    "same_official_split_parent"
                )

    parent_texts.append(parent_text)
    parent_statuses.append(parent_status)


official["parent_text"] = parent_texts

official["parent_status"] = parent_statuses

official["parent_available"] = (
    official["parent_text"].str.len() > 0
)


# ============================================================
# 7. TOKENIZE CURRENT COMMENTS
# ============================================================

print("\nTokenizing current comments...")

official["current_token_length"] = tokenize_lengths(
    tokenizer,
    official["text"].tolist(),
    BATCH_SIZE
)


# ============================================================
# 8. TOKENIZE AVAILABLE PARENT COMMENTS
# ============================================================

print("Tokenizing available parent comments...")

parent_mask = official["parent_available"]

available_parent_texts = official.loc[
    parent_mask,
    "parent_text"
].tolist()

available_parent_lengths = tokenize_lengths(
    tokenizer,
    available_parent_texts,
    BATCH_SIZE
)

official["parent_token_length"] = pd.NA

official.loc[
    parent_mask,
    "parent_token_length"
] = available_parent_lengths

official["parent_token_length"] = pd.to_numeric(
    official["parent_token_length"],
    errors="coerce"
)


# ============================================================
# 9. TOKENIZE CURRENT-PARENT PAIRS
# ============================================================

print("Tokenizing current-parent pairs...")

current_pair_texts = official.loc[
    parent_mask,
    "text"
].tolist()

parent_pair_texts = official.loc[
    parent_mask,
    "parent_text"
].tolist()

pair_lengths = tokenize_pair_lengths(
    tokenizer,
    current_pair_texts,
    parent_pair_texts,
    BATCH_SIZE
)

official["pair_token_length"] = pd.NA

official.loc[
    parent_mask,
    "pair_token_length"
] = pair_lengths

official["pair_token_length"] = pd.to_numeric(
    official["pair_token_length"],
    errors="coerce"
)


# ============================================================
# 10. OVERALL TOKEN-LENGTH SUMMARY
# ============================================================

summary_rows = []

for input_name, column in [

    ("Current comment", "current_token_length"),

    ("Parent comment", "parent_token_length"),

    ("Current + parent pair", "pair_token_length")

]:

    stats = summarize_lengths(
        official[column]
    )

    summary_rows.append({

        "input_type": input_name,

        **stats

    })


overall_summary = pd.DataFrame(
    summary_rows
)

print("\n" + "=" * 65)
print("OVERALL TOKEN-LENGTH SUMMARY")
print("=" * 65)

print(
    overall_summary.to_string(index=False)
)


# ============================================================
# 11. TRUNCATION ANALYSIS
# ============================================================

truncation_rows = []

for input_name, column in [

    ("Current comment", "current_token_length"),

    ("Parent comment", "parent_token_length"),

    ("Current + parent pair", "pair_token_length")

]:

    for max_length in MAX_LENGTHS:

        values = official[column].dropna()

        total = len(values)

        truncated = truncation_count(
            official[column],
            max_length
        )

        percentage = (
            (truncated / total) * 100
            if total > 0
            else 0
        )

        truncation_rows.append({

            "input_type": input_name,

            "max_length": max_length,

            "total_records": total,

            "truncated_records": truncated,

            "truncation_percentage": round(
                percentage,
                2
            )

        })


truncation_summary = pd.DataFrame(
    truncation_rows
)

print("\n" + "=" * 65)
print("TRUNCATION ANALYSIS")
print("=" * 65)

print(
    truncation_summary.to_string(index=False)
)


# ============================================================
# 12. SPLIT-WISE TOKEN-LENGTH ANALYSIS
# ============================================================

split_rows = []

for split, group in official.groupby(
    "official_split"
):

    for input_name, column in [

        ("Current comment", "current_token_length"),

        ("Parent comment", "parent_token_length"),

        ("Current + parent pair", "pair_token_length")

    ]:

        stats = summarize_lengths(
            group[column]
        )

        split_rows.append({

            "split": split,

            "input_type": input_name,

            **stats

        })


split_summary = pd.DataFrame(
    split_rows
)


# ============================================================
# 13. LABEL-WISE TOKEN-LENGTH ANALYSIS
# ============================================================

label_rows = []

for _, row in official.iterrows():

    labels = [
        label.strip()
        for label in row["labels"].split(",")
        if label.strip()
    ]

    for label in labels:

        label_rows.append({

            "label": label,

            "current_token_length": (
                row["current_token_length"]
            ),

            "parent_token_length": (
                row["parent_token_length"]
            ),

            "pair_token_length": (
                row["pair_token_length"]
            ),

            "parent_available": (
                row["parent_available"]
            )

        })


label_records = pd.DataFrame(label_rows)

label_summary_rows = []

for label, group in label_records.groupby("label"):

    for input_name, column in [

        ("Current comment", "current_token_length"),

        ("Parent comment", "parent_token_length"),

        ("Current + parent pair", "pair_token_length")

    ]:

        stats = summarize_lengths(
            group[column]
        )

        label_summary_rows.append({

            "label": label,

            "input_type": input_name,

            **stats

        })


label_summary = pd.DataFrame(
    label_summary_rows
)


# ============================================================
# 14. PARENT AVAILABILITY SUMMARY
# ============================================================

parent_availability_summary = (
    official.groupby(
        ["official_split", "parent_status"]
    )
    .size()
    .reset_index(name="records")
)


# ============================================================
# 15. SAVE RECORD-LEVEL DATA
# ============================================================

record_columns = [

    "id",

    "official_split",

    "labels",

    "text",

    "current_token_length",

    "parent_available",

    "parent_status",

    "parent_text",

    "parent_token_length",

    "pair_token_length"

]

record_path = (
    REPORT_DIR / "bert_token_lengths_record_level.csv"
)

official[record_columns].to_csv(
    record_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. SAVE SUMMARY REPORTS
# ============================================================

reports = {

    "bert_token_length_summary.csv":
        overall_summary,

    "bert_token_truncation_summary.csv":
        truncation_summary,

    "bert_token_length_by_split.csv":
        split_summary,

    "bert_token_length_by_label.csv":
        label_summary,

    "bert_parent_availability.csv":
        parent_availability_summary

}

for filename, dataframe in reports.items():

    dataframe.to_csv(
        REPORT_DIR / filename,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 17. PRINT PARENT AVAILABILITY
# ============================================================

print("\n" + "=" * 65)
print("PARENT AVAILABILITY")
print("=" * 65)

print(
    official["parent_status"]
    .value_counts()
    .to_string()
)


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("TOKEN-LENGTH ANALYSIS COMPLETED")
print("=" * 65)

print("\nReports saved in:", REPORT_DIR)

print("\nGenerated files:")

print("- bert_token_lengths_record_level.csv")

for filename in reports:

    print("-", filename)