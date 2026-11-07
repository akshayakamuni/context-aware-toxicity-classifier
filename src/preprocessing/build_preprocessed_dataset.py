"""
CAD v1.1 - Build Preprocessed Dataset

Purpose:
1. Load official train/dev/test datasets.
2. Load CAD v1.1 metadata.
3. Reconstruct immediate parent comments.
4. Use only parents from the same official split.
5. Convert official labels into multi-hot vectors.
6. Add explicit context availability indicators.
7. Save processed train/dev/test CSV files.
8. Validate records, labels and split integrity.

Important:
- Official labels remain unchanged.
- Original raw data is never modified.
- Missing-context records are preserved.
- No text normalization or truncation is performed here.
"""

from pathlib import Path
from zipfile import ZipFile

import json
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ARCHIVE = (
    PROJECT_ROOT / "datasets" / "raw" / "data.zip"
)

PROCESSED_DIR = (
    PROJECT_ROOT / "datasets" / "processed"
)

REPORT_DIR = (
    PROJECT_ROOT / "reports"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Fixed label order. Do not change this order between experiments.
LABELS = [
    "Neutral",
    "IdentityDirectedAbuse",
    "AffiliationDirectedAbuse",
    "PersonDirectedAbuse",
    "CounterSpeech"
]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """
    Convert missing values to empty strings.

    The string 'NA' is treated as missing because CAD metadata
    uses it for unavailable parent IDs.
    """

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


def read_tsv_from_zip(archive, member):
    """Read one TSV file directly from the raw ZIP archive."""

    with archive.open(member) as file:

        return pd.read_csv(
            file,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8"
        )


def find_metadata_member(archive):
    """Locate the CAD v1.1 metadata TSV."""

    candidates = [
        name
        for name in archive.namelist()
        if Path(name).name.lower() == "cad_v1_1.tsv"
    ]

    if len(candidates) != 1:

        raise RuntimeError(
            "Expected exactly one cad_v1_1.tsv file. "
            f"Found: {candidates}"
        )

    return candidates[0]


def find_split_member(archive, split):
    """Locate one official CAD v1.1 split file."""

    candidates = []

    for name in archive.namelist():

        if not name.lower().endswith(".tsv"):
            continue

        basename = Path(name).name.lower()

        if split not in basename:
            continue

        if (
            "v1_1" not in basename
            and "v1.1" not in basename
        ):
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


def encode_labels(label_string):
    """
    Convert comma-separated official labels into a multi-hot vector.

    Example:
    IdentityDirectedAbuse,PersonDirectedAbuse

    becomes:
    [0, 1, 0, 1, 0]
    """

    original_labels = [
        label.strip()
        for label in label_string.split(",")
        if label.strip()
    ]

    unknown_labels = (
        set(original_labels) - set(LABELS)
    )

    if unknown_labels:

        raise ValueError(
            f"Unknown labels found: {unknown_labels}"
        )

    label_set = set(original_labels)

    return [
        int(label in label_set)
        for label in LABELS
    ]


# ============================================================
# 3. LOAD RAW DATA
# ============================================================

print("=" * 65)
print("CAD v1.1 - PREPROCESSING PIPELINE")
print("=" * 65)

if not RAW_ARCHIVE.exists():

    raise FileNotFoundError(
        f"Dataset archive not found: {RAW_ARCHIVE}"
    )


with ZipFile(RAW_ARCHIVE, "r") as archive:

    metadata_member = find_metadata_member(
        archive
    )

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

        frame["split"] = split

        official_frames.append(frame)


official = pd.concat(
    official_frames,
    ignore_index=True
)

print("\nOfficial records loaded:", len(official))

print("\nOfficial split distribution:")

print(
    official["split"]
    .value_counts()
    .to_string()
)


# ============================================================
# 4. VALIDATE OFFICIAL DATA
# ============================================================

required_official_columns = {
    "id",
    "text",
    "labels",
    "split"
}

missing = (
    required_official_columns
    - set(official.columns)
)

if missing:

    raise ValueError(
        f"Official data is missing columns: {missing}"
    )


official["id"] = official["id"].map(clean_value)

# Preserve the actual text. Do not strip punctuation,
# lowercase, remove URLs or otherwise normalize it.
official["text"] = official["text"].map(
    lambda value: "" if value is None else str(value)
)

official["labels"] = (
    official["labels"].map(clean_value)
)

if official["id"].duplicated().any():

    duplicate_ids = official.loc[
        official["id"].duplicated(keep=False),
        "id"
    ].tolist()

    raise ValueError(
        "Duplicate IDs found across official splits: "
        f"{duplicate_ids[:10]}"
    )

if (official["labels"] == "").any():

    raise ValueError(
        "Some official records have empty labels."
    )


# ============================================================
# 5. PREPARE METADATA
# ============================================================

required_metadata_columns = {
    "info_id",
    "info_id.parent",
    "meta_text"
}

missing = (
    required_metadata_columns
    - set(metadata.columns)
)

if missing:

    raise ValueError(
        f"Metadata is missing columns: {missing}"
    )


metadata["info_id"] = (
    metadata["info_id"].map(clean_value)
)

metadata["info_id.parent"] = (
    metadata["info_id.parent"].map(clean_value)
)

metadata["meta_text"] = (
    metadata["meta_text"].map(
        lambda value: (
            "" if value is None else str(value)
        )
    )
)


# ============================================================
# 6. CHECK REPEATED METADATA IDS
# ============================================================

# The earlier audits found repeated metadata IDs.
# We verify that parent ID and text do not conflict before
# choosing one representative row.

metadata_conflicts = (
    metadata.groupby("info_id")[
        ["info_id.parent", "meta_text"]
    ]
    .nunique(dropna=False)
)

conflicting_ids = metadata_conflicts[
    (metadata_conflicts > 1).any(axis=1)
]

print(
    "\nMetadata IDs with conflicting parent/text values:",
    len(conflicting_ids)
)

if len(conflicting_ids) > 0:

    raise ValueError(
        "Conflicting parent IDs or comment texts were found. "
        "Resolve these before constructing context inputs."
    )


metadata_unique = (
    metadata.drop_duplicates(
        subset=["info_id"],
        keep="first"
    )
    .set_index("info_id")
)


# ============================================================
# 7. BUILD OFFICIAL SPLIT LOOKUP
# ============================================================

official_split_lookup = (
    official.set_index("id")["split"].to_dict()
)


# ============================================================
# 8. RECONSTRUCT IMMEDIATE PARENT CONTEXT
# ============================================================

parent_texts = []
parent_ids = []
parent_statuses = []
context_flags = []

for _, row in official.iterrows():

    comment_id = row["id"]
    current_split = row["split"]

    parent_id = ""
    parent_text = ""

    if comment_id not in metadata_unique.index:

        status = "current_metadata_missing"

    else:

        parent_id = clean_value(
            metadata_unique.loc[
                comment_id,
                "info_id.parent"
            ]
        )

        if not parent_id:

            status = "no_parent_id"

        elif parent_id not in metadata_unique.index:

            status = "parent_missing_from_metadata"

        else:

            candidate_parent_text = str(
                metadata_unique.loc[
                    parent_id,
                    "meta_text"
                ]
            )

            parent_split = (
                official_split_lookup.get(
                    parent_id,
                    ""
                )
            )

            if not candidate_parent_text.strip():

                status = "parent_text_missing"

            elif not parent_split:

                status = "parent_not_in_official_splits"

            elif parent_split != current_split:

                status = "parent_in_different_official_split"

            else:

                parent_text = candidate_parent_text

                status = "same_official_split_parent"

    available = int(bool(parent_text.strip()))

    parent_ids.append(parent_id)
    parent_texts.append(parent_text)
    parent_statuses.append(status)
    context_flags.append(available)


official["parent_id"] = parent_ids

official["parent_text"] = parent_texts

official["parent_status"] = parent_statuses

official["context_available"] = context_flags


# ============================================================
# 9. ENCODE MULTI-LABEL TARGETS
# ============================================================

print("\nEncoding official labels...")

label_vectors = []

for label_string in official["labels"]:

    vector = encode_labels(label_string)

    label_vectors.append(vector)


official["label_vector"] = [
    json.dumps(vector)
    for vector in label_vectors
]


# ============================================================
# 10. ADD LABEL MEMBERSHIP COLUMNS
# ============================================================

# These columns make it easier to analyze each label.
# They are derived from the official label vector.

for index, label in enumerate(LABELS):

    official[f"label_{index}"] = [
        vector[index]
        for vector in label_vectors
    ]


# ============================================================
# 11. BUILD PROCESSED DATAFRAME
# ============================================================

processed_columns = [

    "id",

    "text",

    "labels",

    "label_vector",

    "label_0",

    "label_1",

    "label_2",

    "label_3",

    "label_4",

    "parent_id",

    "parent_text",

    "context_available",

    "parent_status",

    "split"

]

processed = official[
    processed_columns
].copy()


# ============================================================
# 12. VALIDATE PROCESSED DATA
# ============================================================

print("\nValidating processed data...")

assert len(processed) == len(official)

assert processed["id"].is_unique

assert set(processed["split"].unique()) == {
    "train",
    "dev",
    "test"
}

assert processed["context_available"].isin(
    [0, 1]
).all()

assert (
    processed["context_available"]
    == processed["parent_text"].str.strip().ne("").astype(int)
).all()

# Verify that every label vector has five entries.
for vector_string in processed["label_vector"]:

    vector = json.loads(vector_string)

    assert len(vector) == len(LABELS)

    assert all(value in [0, 1] for value in vector)


# Verify each available parent belongs to the same split.
for _, row in processed[
    processed["context_available"] == 1
].iterrows():

    assert (
        official_split_lookup[row["parent_id"]]
        == row["split"]
    )


# ============================================================
# 13. SAVE EACH OFFICIAL SPLIT
# ============================================================

for split in ["train", "dev", "test"]:

    split_df = processed[
        processed["split"] == split
    ].copy()

    output_file = (
        PROCESSED_DIR
        / f"cad_{split}_processed.csv"
    )

    split_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nSaved {split}: {len(split_df)} records"
    )

    print(output_file)


# ============================================================
# 14. SAVE LABEL CONFIGURATION
# ============================================================

label_config = {

    "dataset": "CAD v1.1",

    "labels": LABELS,

    "encoding": "multi-hot",

    "context": "immediate parent comment",

    "context_policy": (
        "Only parent comments from the same official split"
    )

}

config_path = (
    PROCESSED_DIR / "label_configuration.json"
)

config_path.write_text(
    json.dumps(
        label_config,
        indent=4
    ),
    encoding="utf-8"
)


# ============================================================
# 15. GENERATE PREPROCESSING SUMMARY
# ============================================================

summary = {

    "total_records": len(processed),

    "split_counts": (
        processed["split"]
        .value_counts()
        .to_dict()
    ),

    "parent_status_counts": (
        processed["parent_status"]
        .value_counts()
        .to_dict()
    ),

    "context_available_records": int(
        processed["context_available"].sum()
    ),

    "context_missing_records": int(
        (processed["context_available"] == 0).sum()
    ),

    "label_counts": {

        label: int(
            processed[f"label_{index}"].sum()
        )

        for index, label in enumerate(LABELS)

    }

}

summary_path = (
    REPORT_DIR / "preprocessing_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=4
    ),
    encoding="utf-8"
)


# ============================================================
# 16. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("PREPROCESSING SUMMARY")
print("=" * 65)

print("\nTotal records:", len(processed))

print("\nContext availability:")

print(
    processed["parent_status"]
    .value_counts()
    .to_string()
)

print("\nContext available:", summary[
    "context_available_records"
])

print("Context missing:", summary[
    "context_missing_records"
])

print("\nLabel membership counts:")

for label, count in summary["label_counts"].items():

    print(f"{label}: {count}")


print("\n" + "=" * 65)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nProcessed data directory:")
print(PROCESSED_DIR)

print("\nSummary report:")
print(summary_path)