
# ============================================================
# LABEL-WISE CONVERSATION CONTEXT BIAS AUDIT
# Project: Context-Aware Toxicity Classifier
# Dataset: Contextual Abuse Dataset (CAD) v1.1
#
# Purpose:
# 1. Compare conversation context availability across labels.
# 2. Measure ancestor depth for each official label.
# 3. Compare context annotations across labels.
# 4. Check context reconstruction termination reasons.
# 5. Save CSV files and a Markdown report.
#
# IMPORTANT:
# Official split IDs are matched with metadata using info_id.
# Metadata is deduplicated by info_id.
# Only same-split ancestors with non-empty text are counted.
# ============================================================

from pathlib import Path
from collections import defaultdict
from io import BytesIO
import re
import zipfile

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

ARCHIVE_PATH = RAW_DIR / "data.zip"

SPLIT_FILES = {
    "train": "cad_v1_1_train.tsv",
    "dev": "cad_v1_1_dev.tsv",
    "test": "cad_v1_1_test.tsv",
}

METADATA_FILE = "cad_v1_1.tsv"

EXPECTED_SPLIT_COUNTS = {
    "train": 13584,
    "dev": 4526,
    "test": 5307,
}

EXPECTED_TOTAL = 23417

OFFICIAL_LABELS = [
    "Neutral",
    "IdentityDirectedAbuse",
    "AffiliationDirectedAbuse",
    "PersonDirectedAbuse",
    "CounterSpeech",
]


# ============================================================
# 2. GENERAL HELPER FUNCTIONS
# ============================================================

def clean_id(value):
    """Convert IDs into consistent strings."""

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.lower() in {
        "", "nan", "none", "null", "na"
    }:
        return None

    return value


def valid_text(value):
    """Check whether a text field contains usable text."""

    if pd.isna(value):
        return False

    return bool(str(value).strip())


def parse_labels(value):
    """Parse official single-label or multi-label targets."""

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    parts = re.split(r"[|;,]", value)

    labels = []

    for part in parts:
        label = part.strip()

        if label in OFFICIAL_LABELS and label not in labels:
            labels.append(label)

    return labels


def read_tsv_from_zip(archive, filename):
    """Read a TSV file from the CAD archive."""

    matches = [
        name
        for name in archive.namelist()
        if Path(name).name == filename
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename} in archive; "
            f"found {len(matches)}."
        )

    with archive.open(matches[0]) as file:
        return pd.read_csv(
            BytesIO(file.read()),
            sep="\t",
            dtype=str,
            keep_default_na=True,
        )


# ============================================================
# 3. LOAD OFFICIAL TRAIN / DEV / TEST DATA
# ============================================================

def load_official_records(archive):

    records = []

    for split_name, filename in SPLIT_FILES.items():

        df = read_tsv_from_zip(archive, filename)

        required_columns = {
            "id",
            "text",
            "labels",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{filename} missing columns: {missing}"
            )

        for _, row in df.iterrows():

            record_id = clean_id(row["id"])
            labels = parse_labels(row["labels"])

            if record_id is None:
                continue

            if not labels:
                raise ValueError(
                    f"Unrecognized or missing official label "
                    f"for ID {record_id}"
                )

            records.append({
                "id": record_id,
                "split": split_name,
                "labels": labels,
                "text": row["text"],
            })

    official = pd.DataFrame(records)

    if official["id"].duplicated().any():
        raise ValueError(
            "Duplicate IDs found across official split files."
        )

    actual_counts = (
        official["split"].value_counts().to_dict()
    )

    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Split validation failed.\n"
            f"Expected: {EXPECTED_SPLIT_COUNTS}\n"
            f"Actual: {actual_counts}"
        )

    if len(official) != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL} official records, "
            f"found {len(official)}"
        )

    print("\nOFFICIAL DATA LOADED")
    print("--------------------")
    print(f"Total records: {len(official):,}")
    print(official["split"].value_counts().to_string())

    return official


# ============================================================
# 4. LOAD METADATA USING info_id
# ============================================================

def load_metadata(archive):

    metadata = read_tsv_from_zip(
        archive,
        METADATA_FILE
    )

    required_columns = {
        "info_id",
        "info_id.parent",
        "meta_text",
        "annotation_Context",
    }

    missing = required_columns - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Metadata missing columns: {missing}"
        )

    # CRITICAL FIX:
    # Official split IDs correspond to metadata info_id.
    metadata["id_clean"] = metadata["info_id"].map(clean_id)

    metadata["parent_clean"] = (
        metadata["info_id.parent"].map(clean_id)
    )

    metadata = metadata.dropna(
        subset=["id_clean"]
    ).copy()

    # Check duplicate metadata IDs for conflicting parent IDs.
    parent_conflicts = (
        metadata.groupby("id_clean")["parent_clean"]
        .nunique(dropna=True)
    )

    if (parent_conflicts > 1).any():

        conflicting_ids = parent_conflicts[
            parent_conflicts > 1
        ].index.tolist()

        raise ValueError(
            "Conflicting parent IDs found for metadata IDs: "
            f"{conflicting_ids[:10]}"
        )

    # Check conflicting text values.
    metadata["_text_clean"] = (
        metadata["meta_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    text_conflicts = (
        metadata.groupby("id_clean")["_text_clean"]
        .nunique()
    )

    if (text_conflicts > 1).any():

        conflicting_ids = text_conflicts[
            text_conflicts > 1
        ].index.tolist()

        raise ValueError(
            "Conflicting text values found for metadata IDs: "
            f"{conflicting_ids[:10]}"
        )

    # Build one entry per unique metadata ID.
    metadata_index = {}

    for record_id, group in metadata.groupby(
        "id_clean",
        sort=False
    ):

        parent_values = [
            value
            for value in group["parent_clean"].tolist()
            if value is not None
        ]

        parent_id = (
            parent_values[0]
            if parent_values
            else None
        )

        text_values = [
            value
            for value in group["_text_clean"].tolist()
            if value
        ]

        text = (
            text_values[0]
            if text_values
            else None
        )

        context_annotations = set()

        for value in group["annotation_Context"].tolist():

            if valid_text(value):
                context_annotations.add(
                    str(value).strip()
                )

        metadata_index[record_id] = {
            "parent_id": parent_id,
            "text": text,
            "context_annotations": context_annotations,
        }

    print("\nMETADATA LOADED")
    print("----------------")
    print(f"Raw metadata rows: {len(metadata):,}")
    print(f"Unique metadata IDs: {len(metadata_index):,}")

    if len(metadata_index) != 26550:
        raise ValueError(
            "Metadata ID validation failed. "
            f"Expected 26,550 unique IDs, "
            f"found {len(metadata_index):,}. "
            "Check the metadata identifier mapping."
        )

    return metadata_index


# ============================================================
# 5. VALIDATE OFFICIAL ID TO METADATA MATCHING
# ============================================================

def validate_metadata_mapping(official, metadata_index):

    official_ids = set(official["id"])
    metadata_ids = set(metadata_index.keys())

    missing_ids = official_ids - metadata_ids

    print("\nMETADATA MAPPING VALIDATION")
    print("---------------------------")
    print(
        f"Official IDs: {len(official_ids):,}"
    )
    print(
        f"Metadata IDs: {len(metadata_ids):,}"
    )
    print(
        f"Official IDs missing from metadata: "
        f"{len(missing_ids):,}"
    )

    if missing_ids:
        print("Examples:", list(missing_ids)[:10])

        raise ValueError(
            "Official-to-metadata matching failed. "
            "Do not use this audit output."
        )

    print("Metadata mapping validation PASSED.")


# ============================================================
# 6. RECONSTRUCT CONVERSATION DEPTH
# ============================================================

def reconstruct_depth(
    record_id,
    split_name,
    metadata_index,
    official_map,
):

    current_id = record_id

    visited = {record_id}

    depth = 0

    termination_reason = "unknown"

    while True:

        current_metadata = metadata_index.get(
            current_id
        )

        if current_metadata is None:

            termination_reason = (
                "current_record_missing_metadata"
            )

            break

        parent_id = current_metadata["parent_id"]

        if parent_id is None:

            termination_reason = "root_or_no_parent"

            break

        if parent_id in visited:

            termination_reason = "cycle_detected"

            break

        parent_metadata = metadata_index.get(
            parent_id
        )

        if parent_metadata is None:

            termination_reason = (
                "parent_missing_from_metadata"
            )

            break

        parent_split = official_map.get(
            parent_id
        )

        if parent_split is None:

            termination_reason = (
                "parent_excluded_from_official_splits"
            )

            break

        if parent_split != split_name:

            termination_reason = "cross_split_parent"

            break

        if not valid_text(parent_metadata["text"]):

            termination_reason = "parent_text_missing"

            break

        # A valid same-split parent with text.
        depth += 1

        visited.add(parent_id)

        current_id = parent_id

    return depth, termination_reason


# ============================================================
# 7. CONTEXT ANNOTATION CATEGORY
# ============================================================

def get_annotation_category(annotations):

    if not annotations:
        return "NoAnnotation"

    has_current = (
        "CurrentContent" in annotations
    )

    has_previous = (
        "PreviousContent" in annotations
    )

    if has_current and has_previous:
        return "CurrentAndPrevious"

    if has_current:
        return "CurrentOnly"

    if has_previous:
        return "PreviousOnly"

    return "OtherAnnotation"


# ============================================================
# 8. BUILD RECORD-LEVEL AUDIT
# ============================================================

def build_audit_table(
    official,
    metadata_index,
):

    official_map = dict(
        zip(
            official["id"],
            official["split"]
        )
    )

    rows = []

    for _, record in official.iterrows():

        record_id = record["id"]
        split_name = record["split"]

        metadata_record = metadata_index[record_id]

        depth, termination_reason = reconstruct_depth(
            record_id,
            split_name,
            metadata_index,
            official_map,
        )

        annotations = metadata_record[
            "context_annotations"
        ]

        rows.append({

            "id": record_id,

            "split": split_name,

            "labels": record["labels"],

            "depth": depth,

            "has_parent_context": depth >= 1,

            "has_two_ancestors": depth >= 2,

            "has_three_ancestors": depth >= 3,

            "context_annotation": (
                get_annotation_category(annotations)
            ),

            "termination_reason": termination_reason,

        })

    audit = pd.DataFrame(rows)

    if len(audit) != EXPECTED_TOTAL:
        raise ValueError(
            "Record-level audit count validation failed."
        )

    if audit["id"].nunique() != EXPECTED_TOTAL:
        raise ValueError(
            "Record-level audit contains duplicate IDs."
        )

    print("\nRECORD-LEVEL VALIDATION")
    print("-----------------------")
    print(
        f"Official records audited: {len(audit):,}"
    )
    print(
        "Official records missing metadata: 0"
    )

    print(
        "Average ancestor depth:",
        round(audit["depth"].mean(), 4)
    )

    print(
        "Records with at least one ancestor:",
        int(audit["has_parent_context"].sum())
    )

    return audit


# ============================================================
# 9. LABEL-WISE SUMMARY
# ============================================================

def build_label_summary(audit):

    rows = []

    for label in OFFICIAL_LABELS:

        subset = audit[
            audit["labels"].apply(
                lambda labels: label in labels
            )
        ]

        total = len(subset)

        rows.append({

            "label": label,

            "records": total,

            "percentage_of_all_official_records":
                round(
                    total / len(audit) * 100,
                    3
                ),

            "mean_ancestor_depth":
                round(
                    subset["depth"].mean(),
                    4
                ),

            "median_ancestor_depth":
                round(
                    subset["depth"].median(),
                    4
                ),

            "records_with_parent_context":
                int(
                    subset["has_parent_context"].sum()
                ),

            "parent_context_percentage":
                round(
                    subset["has_parent_context"].mean()
                    * 100,
                    3
                ),

            "records_with_two_ancestors":
                int(
                    subset["has_two_ancestors"].sum()
                ),

            "two_ancestor_percentage":
                round(
                    subset["has_two_ancestors"].mean()
                    * 100,
                    3
                ),

            "records_with_three_ancestors":
                int(
                    subset["has_three_ancestors"].sum()
                ),

            "three_ancestor_percentage":
                round(
                    subset["has_three_ancestors"].mean()
                    * 100,
                    3
                ),

        })

    return pd.DataFrame(rows)


# ============================================================
# 10. LABEL-WISE DEPTH DISTRIBUTION
# ============================================================

def build_depth_distribution(audit):

    rows = []

    for label in OFFICIAL_LABELS:

        subset = audit[
            audit["labels"].apply(
                lambda labels: label in labels
            )
        ]

        total = len(subset)

        counts = (
            subset["depth"]
            .value_counts()
            .sort_index()
        )

        for depth, count in counts.items():

            rows.append({

                "label": label,

                "depth": int(depth),

                "records": int(count),

                "percentage_within_label":
                    round(
                        count / total * 100,
                        3
                    ),

            })

    return pd.DataFrame(rows)


# ============================================================
# 11. LABEL-WISE CONTEXT ANNOTATION DISTRIBUTION
# ============================================================

def build_annotation_distribution(audit):

    rows = []

    for label in OFFICIAL_LABELS:

        subset = audit[
            audit["labels"].apply(
                lambda labels: label in labels
            )
        ]

        total = len(subset)

        counts = (
            subset["context_annotation"]
            .value_counts()
        )

        for annotation, count in counts.items():

            rows.append({

                "label": label,

                "context_annotation": annotation,

                "records": int(count),

                "percentage_within_label":
                    round(
                        count / total * 100,
                        3
                    ),

            })

    return pd.DataFrame(rows)


# ============================================================
# 12. TERMINATION REASON DISTRIBUTION
# ============================================================

def build_termination_summary(audit):

    return (
        audit.groupby(
            [
                "split",
                "termination_reason"
            ]
        )
        .size()
        .reset_index(name="records")
        .sort_values(
            [
                "split",
                "records"
            ],
            ascending=[
                True,
                False
            ]
        )
    )


# ============================================================
# 13. GENERATE MARKDOWN REPORT
# ============================================================

def generate_report(
    audit,
    summary,
    depth_distribution,
    annotation_distribution,
    termination_summary,
):

    lines = [

        "# Label-Wise Conversation Context Audit",

        "",

        "## 1. Purpose",

        "",

        "This audit examines whether conversation context "
        "availability and reconstructable ancestor depth "
        "differ across the five official CAD v1.1 labels.",

        "",

        "Only same-split ancestors with non-empty text "
        "are counted.",

        "",

        "## 2. Validation",

        "",

        f"- Official records audited: {len(audit):,}",

        f"- Unique official IDs: {audit['id'].nunique():,}",

        f"- Records with at least one ancestor: "
        f"{int(audit['has_parent_context'].sum()):,}",

        f"- Records with at least two ancestors: "
        f"{int(audit['has_two_ancestors'].sum()):,}",

        f"- Records with at least three ancestors: "
        f"{int(audit['has_three_ancestors'].sum()):,}",

        "",

        "## 3. Label-wise summary",

        "",

        summary.to_markdown(index=False),

        "",

        "## 4. Label-wise depth distribution",

        "",

        depth_distribution.to_markdown(index=False),

        "",

        "## 5. Context annotation distribution",

        "",

        annotation_distribution.to_markdown(index=False),

        "",

        "## 6. Chain termination reasons",

        "",

        termination_summary.to_markdown(index=False),

        "",

        "## 7. Interpretation",

        "",

        "- Compare parent-context availability across labels.",

        "- Compare average and median conversation depth.",

        "- Check whether rare labels have less available context.",

        "- Context annotation is not equivalent to actual "
        "parent-text availability.",

        "- Multi-label records contribute to each label "
        "they carry.",

        "- These descriptive results do not establish causation.",

        "",

        "## 8. Research decisions",

        "",

        "- Preserve records with missing context.",

        "- Do not fabricate missing conversation text.",

        "- Use only same-split ancestor text.",

        "- Never use ancestor labels as model inputs.",

        "- Evaluate context-available and context-unavailable "
        "subsets separately.",

        "- Use identical official splits for all context experiments.",

    ]

    return "\n".join(lines)


# ============================================================
# 14. MAIN EXECUTION
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not ARCHIVE_PATH.exists():

        raise FileNotFoundError(
            f"CAD archive not found: {ARCHIVE_PATH}"
        )

    print("=" * 65)
    print("LABEL-WISE CONVERSATION CONTEXT AUDIT")
    print("=" * 65)

    # Load data.
    with zipfile.ZipFile(
        ARCHIVE_PATH,
        "r"
    ) as archive:

        official = load_official_records(
            archive
        )

        metadata_index = load_metadata(
            archive
        )

    # Validate metadata mapping before reconstruction.
    validate_metadata_mapping(
        official,
        metadata_index
    )

    # Reconstruct conversation histories.
    audit = build_audit_table(
        official,
        metadata_index
    )

    # Generate summaries.
    summary = build_label_summary(
        audit
    )

    depth_distribution = build_depth_distribution(
        audit
    )

    annotation_distribution = (
        build_annotation_distribution(audit)
    )

    termination_summary = (
        build_termination_summary(audit)
    )

    # Save CSV outputs.
    outputs = {

        "label_context_summary.csv":
            summary,

        "label_depth_distribution.csv":
            depth_distribution,

        "label_context_annotations.csv":
            annotation_distribution,

        "label_context_termination.csv":
            termination_summary,

        "label_context_record_level.csv":
            audit,

    }

    for filename, dataframe in outputs.items():

        dataframe.to_csv(
            REPORT_DIR / filename,
            index=False
        )

    # Save Markdown report.
    report = generate_report(
        audit,
        summary,
        depth_distribution,
        annotation_distribution,
        termination_summary,
    )

    report_path = (
        REPORT_DIR / "LABEL_CONTEXT_AUDIT.md"
    )

    report_path.write_text(
        report,
        encoding="utf-8"
    )

    # Display results.
    print("\n" + "=" * 65)
    print("LABEL-WISE CONTEXT SUMMARY")
    print("=" * 65)

    print(
        summary.to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("LABEL-WISE DEPTH DISTRIBUTION")
    print("=" * 65)

    print(
        depth_distribution.to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("CONTEXT ANNOTATION DISTRIBUTION")
    print("=" * 65)

    print(
        annotation_distribution.to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("TERMINATION REASONS")
    print("=" * 65)

    print(
        termination_summary.to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(f"\nReports saved in: {REPORT_DIR}")

    for filename in outputs:
        print(f"  - {filename}")

    print("  - LABEL_CONTEXT_AUDIT.md")


if __name__ == "__main__":
    main()
