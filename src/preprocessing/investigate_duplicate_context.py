"""
CAD v1.1 - Contextual Investigation of Cross-Split Duplicate Comments

Purpose:
1. Load the 78 cross-split duplicate text groups.
2. Retrieve their official labels and split assignments.
3. Match each duplicate comment to CAD metadata.
4. Retrieve its parent comment, thread information and context annotation.
5. Identify label-conflict groups and summarize the findings.

Important:
- Does not modify raw data.
- Does not change official labels.
- Uses metadata only for investigation.
"""

from pathlib import Path
from zipfile import ZipFile
import ast
import io

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ARCHIVE = PROJECT_ROOT / "datasets" / "raw" / "data.zip"

DUPLICATE_FILE = (
    PROJECT_ROOT
    / "reports"
    / "detailed_cross_split_duplicates.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def read_tsv_from_zip(archive, member):
    """Read one TSV file directly from the ZIP archive."""

    with archive.open(member) as file:
        return pd.read_csv(
            file,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8"
        )


def find_metadata_member(archive):
    """Find the CAD v1.1 metadata TSV inside the ZIP."""

    candidates = [
        name for name in archive.namelist()
        if Path(name).name.lower() == "cad_v1_1.tsv"
    ]

    if not candidates:
        raise FileNotFoundError(
            "Could not find cad_v1_1.tsv inside data.zip"
        )

    if len(candidates) > 1:
        print("Multiple metadata candidates found:")
        for candidate in candidates:
            print("  ", candidate)

    return candidates[0]


def find_official_split_members(archive):
    """
    Find official train, dev and test TSV files.

    A valid split file must contain:
    id, text and labels columns.
    """

    split_members = {}

    for split in ["train", "dev", "test"]:

        candidates = []

        for name in archive.namelist():

            if not name.lower().endswith(".tsv"):
                continue

            basename = Path(name).name.lower()

            if split not in basename:
                continue

            try:
                with archive.open(name) as file:
                    columns = pd.read_csv(
                        file,
                        sep="\t",
                        nrows=0
                    ).columns.tolist()

                required = {"id", "text", "labels"}

                if required.issubset(set(columns)):
                    candidates.append(name)

            except Exception:
                continue

        if not candidates:
            raise FileNotFoundError(
                f"Could not find official {split} TSV file."
            )

        # Prefer filenames explicitly identifying CAD v1.1.
        preferred = [
            name for name in candidates
            if "v1_1" in Path(name).name.lower()
            or "v1.1" in Path(name).name.lower()
        ]

        if len(preferred) == 1:
            split_members[split] = preferred[0]

        elif len(candidates) == 1:
            split_members[split] = candidates[0]

        else:
            raise RuntimeError(
                f"Multiple possible {split} files found:\n"
                + "\n".join(candidates)
                + "\nPlease inspect the archive filenames."
            )

    return split_members


def parse_list(value):
    """Safely convert a CSV string representation of a list."""

    if isinstance(value, list):
        return value

    if not isinstance(value, str) or not value.strip():
        return []

    try:
        result = ast.literal_eval(value)

        if isinstance(result, list):
            return result

    except (ValueError, SyntaxError):
        pass

    return []


def clean_value(value):
    """Represent empty metadata values consistently."""

    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in {"nan", "none", "null"}:
        return ""

    return value


# ============================================================
# 3. CHECK INPUT FILES
# ============================================================

print("=" * 65)
print("CAD v1.1 - DUPLICATE CONTEXT INVESTIGATION")
print("=" * 65)

if not RAW_ARCHIVE.exists():
    raise FileNotFoundError(
        f"Dataset archive not found: {RAW_ARCHIVE}"
    )

if not DUPLICATE_FILE.exists():
    raise FileNotFoundError(
        f"Duplicate report not found: {DUPLICATE_FILE}"
    )


# ============================================================
# 4. LOAD DUPLICATE GROUPS
# ============================================================

duplicates = pd.read_csv(
    DUPLICATE_FILE,
    dtype=str,
    keep_default_na=False
)

required_duplicate_columns = {
    "text",
    "record_ids"
}

if not required_duplicate_columns.issubset(
    set(duplicates.columns)
):
    raise ValueError(
        "Duplicate CSV is missing required columns: "
        "text and/or record_ids"
    )

duplicates["record_ids_list"] = (
    duplicates["record_ids"].apply(parse_list)
)

print(
    "\nUnique cross-split duplicate groups:",
    len(duplicates)
)


# ============================================================
# 5. LOAD OFFICIAL SPLITS AND METADATA
# ============================================================

with ZipFile(RAW_ARCHIVE, "r") as archive:

    metadata_member = find_metadata_member(archive)

    split_members = find_official_split_members(archive)

    print("\nMetadata file:", metadata_member)

    print("\nOfficial split files:")
    for split, member in split_members.items():
        print(f"{split}: {member}")

    # Load metadata.
    metadata = read_tsv_from_zip(
        archive,
        metadata_member
    )

    # Load official train/dev/test labels.
    official_frames = []

    for split, member in split_members.items():

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


# ============================================================
# 6. VALIDATE REQUIRED METADATA COLUMNS
# ============================================================

required_metadata_columns = [
    "info_id",
    "info_id.parent",
    "info_thread.id",
    "info_order",
    "info_subreddit",
    "annotation_Context",
    "meta_text",
    "split"
]

missing_columns = [
    column
    for column in required_metadata_columns
    if column not in metadata.columns
]

if missing_columns:
    raise ValueError(
        "Metadata is missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# 7. PREPARE OFFICIAL RECORD LOOKUP
# ============================================================

official["id"] = official["id"].map(clean_value)

official_lookup = official[
    ["id", "text", "labels", "official_split"]
].copy()

official_lookup = official_lookup.drop_duplicates(
    subset=["id"],
    keep="first"
)

official_lookup = official_lookup.set_index("id")


# ============================================================
# 8. PREPARE METADATA LOOKUP
# ============================================================

metadata["info_id"] = metadata["info_id"].map(clean_value)

# The metadata can contain repeated info_id values.
# We check whether important fields conflict before selecting
# one representative row per info_id.

metadata_fields = [
    "info_id.parent",
    "info_thread.id",
    "info_order",
    "info_subreddit",
    "annotation_Context",
    "meta_text",
    "split"
]

conflict_counts = (
    metadata.groupby("info_id")[metadata_fields]
    .nunique(dropna=False)
)

conflicting_metadata_ids = conflict_counts[
    (conflict_counts > 1).any(axis=1)
]

print(
    "\nMetadata IDs with conflicting values in "
    "investigation fields:",
    len(conflicting_metadata_ids)
)

# Earlier dataset audit found repeated metadata IDs.
# We preserve one representative row only after reporting
# whether important fields conflict.

metadata_unique = (
    metadata.drop_duplicates(
        subset=["info_id"],
        keep="first"
    )
    .set_index("info_id")
)


# ============================================================
# 9. BUILD RECORD-LEVEL INVESTIGATION
# ============================================================

records = []

for _, duplicate_row in duplicates.iterrows():

    comment_text = duplicate_row["text"]

    record_ids = duplicate_row["record_ids_list"]

    for record_id in record_ids:

        record_id = clean_value(record_id)

        # Retrieve official label and split.
        if record_id in official_lookup.index:

            official_row = official_lookup.loc[record_id]

            official_label = clean_value(
                official_row["labels"]
            )

            official_split = clean_value(
                official_row["official_split"]
            )

            official_text = clean_value(
                official_row["text"]
            )

        else:

            official_label = ""
            official_split = ""
            official_text = ""

        # Retrieve current comment metadata.
        if record_id in metadata_unique.index:

            current_meta = metadata_unique.loc[record_id]

            parent_id = clean_value(
                current_meta["info_id.parent"]
            )

            thread_id = clean_value(
                current_meta["info_thread.id"]
            )

            order = clean_value(
                current_meta["info_order"]
            )

            subreddit = clean_value(
                current_meta["info_subreddit"]
            )

            context_annotation = clean_value(
                current_meta["annotation_Context"]
            )

            metadata_text = clean_value(
                current_meta["meta_text"]
            )

            metadata_split = clean_value(
                current_meta["split"]
            )

        else:

            parent_id = ""
            thread_id = ""
            order = ""
            subreddit = ""
            context_annotation = ""
            metadata_text = ""
            metadata_split = ""

        # Retrieve parent comment metadata.
        parent_text = ""
        parent_thread_id = ""
        parent_split = ""
        parent_official_split = ""
        parent_found = False

        if parent_id and parent_id in metadata_unique.index:

            parent_meta = metadata_unique.loc[parent_id]

            parent_text = clean_value(
                parent_meta["meta_text"]
            )

            parent_thread_id = clean_value(
                parent_meta["info_thread.id"]
            )

            parent_split = clean_value(
                parent_meta["split"]
            )

            parent_found = bool(parent_text)

            if parent_id in official_lookup.index:

                parent_official_split = clean_value(
                    official_lookup.loc[
                        parent_id,
                        "official_split"
                    ]
                )

        # Determine parent availability category.
        if not parent_id:

            parent_status = "no_parent_id"

        elif parent_id not in metadata_unique.index:

            parent_status = "parent_missing_from_metadata"

        elif not parent_text:

            parent_status = "parent_text_missing"

        elif parent_official_split == official_split:

            parent_status = "same_official_split_parent"

        elif parent_official_split:

            parent_status = "parent_in_different_official_split"

        else:

            parent_status = "parent_not_in_official_splits"

        records.append({

            "duplicate_text": comment_text,

            "record_id": record_id,

            "official_label": official_label,

            "official_split": official_split,

            "official_text_matches_duplicate": (
                official_text == comment_text
            ),

            "metadata_text_matches_official": (
                metadata_text == official_text
            ),

            "parent_id": parent_id,

            "parent_text": parent_text,

            "parent_found": parent_found,

            "parent_status": parent_status,

            "thread_id": thread_id,

            "parent_thread_id": parent_thread_id,

            "thread_matches_parent": (
                bool(thread_id)
                and bool(parent_thread_id)
                and thread_id == parent_thread_id
            ),

            "conversation_order": order,

            "subreddit": subreddit,

            "context_annotation": context_annotation,

            "metadata_split": metadata_split

        })


record_df = pd.DataFrame(records)


# ============================================================
# 10. BUILD GROUP-LEVEL SUMMARY
# ============================================================

group_rows = []

for text, group in record_df.groupby(
    "duplicate_text",
    dropna=False
):

    labels = sorted(
        set(
            label
            for label in group["official_label"]
            if label
        )
    )

    splits = sorted(
        set(
            split
            for split in group["official_split"]
            if split
        )
    )

    parent_ids = sorted(
        set(
            parent
            for parent in group["parent_id"]
            if parent
        )
    )

    thread_ids = sorted(
        set(
            thread
            for thread in group["thread_id"]
            if thread
        )
    )

    group_rows.append({

        "duplicate_text": text,

        "text_length": len(text),

        "record_count": len(group),

        "record_ids": ", ".join(
            group["record_id"].tolist()
        ),

        "official_labels": " | ".join(labels),

        "official_splits": " | ".join(splits),

        "different_labels": len(labels) > 1,

        "different_parent_ids": len(parent_ids) > 1,

        "different_threads": len(thread_ids) > 1,

        "unique_parent_count": len(parent_ids),

        "unique_thread_count": len(thread_ids),

        "records_with_parent_text": int(
            group["parent_found"].sum()
        ),

        "records_without_parent_text": int(
            (~group["parent_found"]).sum()
        ),

        "parent_statuses": " | ".join(
            sorted(set(group["parent_status"]))
        ),

        "context_annotations": " | ".join(
            sorted(
                set(
                    value
                    for value in group["context_annotation"]
                    if value
                )
            )
        )

    })


group_df = pd.DataFrame(group_rows)


# ============================================================
# 11. EXTRACT LABEL-CONFLICT GROUPS
# ============================================================

conflict_df = group_df[
    group_df["different_labels"]
].copy()

conflict_records = record_df[
    record_df["duplicate_text"].isin(
        conflict_df["duplicate_text"]
    )
].copy()


# ============================================================
# 12. VALIDATION AND SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("INVESTIGATION SUMMARY")
print("=" * 65)

print(
    "\nDuplicate groups investigated:",
    len(group_df)
)

print(
    "Individual duplicate records:",
    len(record_df)
)

print(
    "Groups with different official labels:",
    len(conflict_df)
)

print(
    "Records with parent text available:",
    int(record_df["parent_found"].sum())
)

print("\nParent status distribution:")

print(
    record_df["parent_status"]
    .value_counts()
    .to_string()
)

print("\nContext annotation distribution:")

print(
    record_df["context_annotation"]
    .replace("", "[No annotation]")
    .value_counts()
    .to_string()
)

print("\nLabel-conflict groups:")

if conflict_df.empty:

    print("No label-conflict groups found.")

else:

    for _, row in conflict_df.iterrows():

        print("\nComment:", repr(row["duplicate_text"]))

        print("Labels:", row["official_labels"])

        print("Splits:", row["official_splits"])

        print("Different parents:", row["different_parent_ids"])

        print("Different threads:", row["different_threads"])


# ============================================================
# 13. SAVE REPORTS
# ============================================================

record_path = (
    REPORT_DIR / "duplicate_context_records.csv"
)

group_path = (
    REPORT_DIR / "duplicate_context_groups.csv"
)

conflict_path = (
    REPORT_DIR / "duplicate_context_label_conflicts.csv"
)

conflict_records_path = (
    REPORT_DIR / "duplicate_context_conflict_records.csv"
)

record_df.to_csv(
    record_path,
    index=False,
    encoding="utf-8-sig"
)

group_df.to_csv(
    group_path,
    index=False,
    encoding="utf-8-sig"
)

conflict_df.to_csv(
    conflict_path,
    index=False,
    encoding="utf-8-sig"
)

conflict_records.to_csv(
    conflict_records_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. GENERATE MARKDOWN REPORT
# ============================================================

report_path = (
    REPORT_DIR / "DUPLICATE_CONTEXT_INVESTIGATION.md"
)

lines = [

    "# CAD v1.1 - Duplicate Context Investigation",

    "",

    "## Overview",

    f"- Duplicate text groups: {len(group_df)}",

    f"- Individual duplicate records: {len(record_df)}",

    f"- Groups with different official labels: {len(conflict_df)}",

    "",

    "## Parent Comment Availability",

    record_df["parent_status"]
    .value_counts()
    .to_string(),

    "",

    "## Context Annotation Distribution",

    record_df["context_annotation"]
    .replace("", "[No annotation]")
    .value_counts()
    .to_string(),

    "",

    "## Label Conflict Groups",

]

for _, row in conflict_df.iterrows():

    lines.extend([

        f"### {repr(row['duplicate_text'])}",

        f"- Labels: {row['official_labels']}",

        f"- Splits: {row['official_splits']}",

        f"- Different parent IDs: {row['different_parent_ids']}",

        f"- Different threads: {row['different_threads']}",

        f"- Parent statuses: {row['parent_statuses']}",

        f"- Context annotations: {row['context_annotations']}",

        ""

    ])


lines.extend([

    "## Important Limitations",

    "",

    "- Different labels do not automatically indicate annotation errors.",

    "- Parent text availability does not guarantee complete conversation history.",

    "- Metadata context annotations are not the official supervised labels.",

    "- This investigation does not modify or remove any dataset records.",

    "- Context relationships must be interpreted carefully before model use.",

])


report_path.write_text(
    "\n".join(lines),
    encoding="utf-8"
)


# ============================================================
# 15. FINISH
# ============================================================

print("\n" + "=" * 65)
print("INVESTIGATION COMPLETED")
print("=" * 65)

print("\nReports saved:")

for path in [
    record_path,
    group_path,
    conflict_path,
    conflict_records_path,
    report_path
]:

    print(path)