
from pathlib import Path
from collections import Counter, defaultdict
import zipfile
import csv


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
ZIP_PATH = RAW_DIR / "data.zip"

METADATA_FILE = "data/cad_v1_1.tsv"

OFFICIAL_SPLITS = {
    "train": "data/cad_v1_1_train.tsv",
    "dev": "data/cad_v1_1_dev.tsv",
    "test": "data/cad_v1_1_test.tsv"
}

SPLIT_NAMES = {"train", "dev", "test"}

SAMPLE_THREAD_COUNT = 2
SAMPLE_MISMATCH_COUNT = 10


# ============================================================
# SHARED HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_section(title):
    print("\n" + "-" * 65)
    print(title)
    print("-" * 65)


def read_tsv(archive, filename):
    """Read a TSV file from the CAD ZIP archive."""

    with archive.open(filename) as file:
        reader = csv.DictReader(
            (line.decode("utf-8-sig") for line in file),
            delimiter="\t"
        )
        return list(reader)


def parse_labels(value):
    if not value or value.strip() in ("", "NA"):
        return set()

    return {
        label.strip()
        for label in value.split(",")
        if label.strip()
    }


def clean_value(value):
    if value is None or value == "NA":
        return None
    return value


# ============================================================
# 1. DATASET INVENTORY
# ============================================================

def audit_dataset_inventory():

    print_header("1. DATASET INVENTORY")

    if not RAW_DIR.exists():
        print("Raw dataset directory does not exist:", RAW_DIR)
        return

    for path in sorted(RAW_DIR.rglob("*")):

        if path.is_file():
            relative_path = path.relative_to(RAW_DIR)

            print(
                f"{relative_path} "
                f"({path.stat().st_size / (1024 * 1024):.2f} MB)"
            )


# ============================================================
# 2. CAD ARCHIVE INSPECTION
# ============================================================

def audit_archive(archive):

    print_header("2. CAD ARCHIVE INSPECTION")

    members = archive.namelist()

    print("Total files in archive:", len(members))

    print_section("Relevant CAD files")

    for member in members:

        if (
            "cad_v1" in member.lower()
            or "codebook" in member.lower()
            or "readme" in member.lower()
            or "errata" in member.lower()
        ):
            print(member)

    print_section("Archive integrity")

    bad_file = archive.testzip()

    if bad_file is None:
        print("Archive integrity: PASSED")
    else:
        print("Archive integrity: FAILED")
        print("First corrupted file:", bad_file)


# ============================================================
# 3. OFFICIAL TSV INSPECTION
# ============================================================

def audit_tsv_structure(archive):

    print_header("3. OFFICIAL TSV STRUCTURE")

    files = [
        *OFFICIAL_SPLITS.values(),
        METADATA_FILE
    ]

    for filename in files:

        rows = read_tsv(archive, filename)

        print_section(filename)

        print("Number of rows:", len(rows))
        print("Number of columns:", len(rows[0]) if rows else 0)

        if rows:
            print("Columns:")
            print(", ".join(rows[0].keys()))

        if filename == METADATA_FILE and rows:

            print("\nMetadata split distribution:")

            split_counts = Counter(
                row["split"] for row in rows
            )

            for split, count in split_counts.most_common():
                print(f"{split}: {count}")


# ============================================================
# 4. DUPLICATE ID AUDIT
# ============================================================

def audit_duplicate_ids(metadata):

    print_header("4. DUPLICATE ID AUDIT")

    grouped = defaultdict(list)

    for row in metadata:
        grouped[row["info_id"]].append(row)

    duplicate_groups = {
        comment_id: rows
        for comment_id, rows in grouped.items()
        if len(rows) > 1
    }

    print("Total metadata rows:", len(metadata))
    print("Unique IDs:", len(grouped))
    print("Repeated IDs:", len(duplicate_groups))

    extra_rows = sum(
        len(rows) - 1
        for rows in duplicate_groups.values()
    )

    print("Extra duplicate rows:", extra_rows)

    fields = [
        "meta_text",
        "annotation_Primary",
        "annotation_Secondary",
        "annotation_Context",
        "split",
        "info_id.parent",
        "info_thread.id",
        "info_order"
    ]

    print_section("Duplicate groups with different values")

    for field in fields:

        different_count = sum(
            len({row.get(field) for row in rows}) > 1
            for rows in duplicate_groups.values()
        )

        print(f"{field}: {different_count}")

    print_section("Important duplicate statistics")

    print(
        "Groups with multiple splits:",
        sum(
            len({row["split"] for row in rows}) > 1
            for rows in duplicate_groups.values()
        )
    )

    print(
        "Groups with multiple primary labels:",
        sum(
            len({row["annotation_Primary"] for row in rows}) > 1
            for rows in duplicate_groups.values()
        )
    )

    print(
        "Groups with different text:",
        sum(
            len({row["meta_text"] for row in rows}) > 1
            for rows in duplicate_groups.values()
        )
    )

    print(
        "Groups with different context annotations:",
        sum(
            len({row["annotation_Context"] for row in rows}) > 1
            for rows in duplicate_groups.values()
        )
    )

    print_section("Sample duplicate IDs")

    for comment_id, rows in list(duplicate_groups.items())[:5]:

        print("\nID:", comment_id)

        for row in rows:
            print(
                "Primary:",
                row["annotation_Primary"],
                "| Secondary:",
                row["annotation_Secondary"],
                "| Context:",
                row["annotation_Context"]
            )


# ============================================================
# 5. MULTI-LABEL DISTRIBUTION IN METADATA
# ============================================================

def audit_metadata_labels(metadata):

    print_header("5. METADATA MULTI-LABEL DISTRIBUTION")

    grouped_labels = defaultdict(set)
    grouped_context = defaultdict(set)

    for row in metadata:

        comment_id = row["info_id"]

        label = row["annotation_Primary"]

        if label != "NA":
            grouped_labels[comment_id].add(label)

        context = row["annotation_Context"]

        if context not in ("NA", "", None):
            grouped_context[comment_id].add(context)

    label_counts = Counter()

    for labels in grouped_labels.values():
        label_counts.update(labels)

    combination_counts = Counter(
        tuple(sorted(labels))
        for labels in grouped_labels.values()
    )

    print("Total unique comments:", len(grouped_labels))

    single_label = sum(
        len(labels) == 1
        for labels in grouped_labels.values()
    )

    multi_label = sum(
        len(labels) > 1
        for labels in grouped_labels.values()
    )

    print("Single-label comments:", single_label)
    print("Multi-label comments:", multi_label)

    print_section("Individual label counts")

    for label, count in label_counts.most_common():
        print(f"{label}: {count}")

    print_section("Most common label combinations")

    for combination, count in combination_counts.most_common(20):
        print(f"{combination}: {count}")

    print_section("Context annotation combinations")

    context_counts = Counter(
        tuple(sorted(contexts))
        for contexts in grouped_context.values()
    )

    for combination, count in context_counts.most_common():
        print(f"{combination}: {count}")

    print_section("Unique comments per split")

    split_ids = defaultdict(set)

    for row in metadata:
        split_ids[row["split"]].add(row["info_id"])

    for split, ids in split_ids.items():
        print(f"{split}: {len(ids)}")


# ============================================================
# 6. CONTEXT AVAILABILITY AUDIT
# ============================================================

def audit_context_availability(metadata):

    print_header("6. CONTEXT AVAILABILITY AUDIT")

    id_to_splits = defaultdict(set)

    for row in metadata:
        id_to_splits[row["info_id"]].add(row["split"])

    no_parent = 0
    parent_available = 0
    parent_missing = 0

    same_split = 0
    cross_split = 0
    unresolved_parent = 0

    checked_relationships = set()

    for row in metadata:

        child_id = row["info_id"]
        parent_id = clean_value(row["info_id.parent"])
        child_split = row["split"]

        if parent_id is None:
            no_parent += 1
            continue

        if parent_id not in id_to_splits:
            parent_missing += 1
            unresolved_parent += 1
            continue

        parent_available += 1

        # Avoid counting repeated annotation rows as
        # independent parent-child relationships.
        relationship = (
            child_id,
            parent_id,
            child_split
        )

        if relationship in checked_relationships:
            continue

        checked_relationships.add(relationship)

        parent_splits = id_to_splits[parent_id]

        if child_split in parent_splits:
            same_split += 1
        else:
            cross_split += 1

    print("Total metadata rows:", len(metadata))
    print("Unique comment IDs:", len(id_to_splits))

    print("Rows with no parent:", no_parent)
    print("Rows with parent available:", parent_available)
    print("Rows with missing parent:", parent_missing)

    print_section("Parent-child split consistency")

    print("Same split relationships:", same_split)
    print("Cross split relationships:", cross_split)
    print("Unresolved parent references:", unresolved_parent)

    print_section("Split distribution")

    split_counts = Counter(row["split"] for row in metadata)

    for split, count in split_counts.most_common():
        print(f"{split}: {count}")


# ============================================================
# 7. CONVERSATION AND THREAD INSPECTION
# ============================================================

def audit_threads(metadata):

    print_header("7. CONVERSATION AND THREAD INSPECTION")

    threads = defaultdict(list)

    for row in metadata:

        thread_id = clean_value(row["info_thread.id"])

        if thread_id is not None:
            threads[thread_id].append(row)

    print("Total unique threads:", len(threads))

    thread_sizes = Counter(
        len(rows) for rows in threads.values()
    )

    print_section("Thread size distribution")

    for size, count in thread_sizes.most_common(10):
        print(f"{size} records: {count} threads")

    print_section("Sample conversation threads")

    selected_threads = list(threads.items())[:SAMPLE_THREAD_COUNT]

    for thread_id, rows in selected_threads:

        rows.sort(
            key=lambda row: (
                row["info_order"] or ""
            )
        )

        print("\nTHREAD ID:", thread_id)
        print("Number of records:", len(rows))

        for row in rows[:10]:

            print("\nComment ID:", row["info_id"])
            print("Parent ID:", row["info_id.parent"])
            print("Order:", row["info_order"])
            print("Primary label:", row["annotation_Primary"])
            print("Context:", row["annotation_Context"])
            print("Text:", (row["meta_text"] or "")[:300])


# ============================================================
# 8. CONTEXT ANNOTATION INSPECTION
# ============================================================

def audit_context_examples(metadata):

    print_header("8. CONTEXT ANNOTATION EXAMPLES")

    examples = defaultdict(list)

    for row in metadata:

        context = row["annotation_Context"]

        if context in ("CurrentContent", "PreviousContent"):
            examples[context].append(row)

    for context, rows in examples.items():

        print_section(context)

        print("Number of metadata rows:", len(rows))

        for row in rows[:5]:

            print("\nComment ID:", row["info_id"])
            print("Parent ID:", row["info_id.parent"])
            print("Primary label:", row["annotation_Primary"])
            print("Text:", (row["meta_text"] or "")[:300])


# ============================================================
# 9. OFFICIAL LABEL VERIFICATION
# ============================================================

def audit_official_label_verification(archive, metadata):

    print_header("9. OFFICIAL LABEL VERIFICATION")

    metadata_labels = defaultdict(set)
    metadata_splits = defaultdict(set)

    for row in metadata:

        comment_id = row["info_id"]
        label = row["annotation_Primary"]

        if label != "NA":
            metadata_labels[comment_id].add(label)

        metadata_splits[comment_id].add(row["split"])

    total_mismatches = 0

    for split, filename in OFFICIAL_SPLITS.items():

        official_rows = read_tsv(archive, filename)

        official_labels = {
            row["id"]: parse_labels(row["labels"])
            for row in official_rows
        }

        metadata_ids = {
            comment_id
            for comment_id, splits in metadata_splits.items()
            if splits == {split}
        }

        official_ids = set(official_labels)

        missing_from_metadata = official_ids - metadata_ids
        missing_from_official = metadata_ids - official_ids

        common_ids = official_ids & metadata_ids

        mismatches = []

        for comment_id in common_ids:

            if metadata_labels[comment_id] != official_labels[comment_id]:

                mismatches.append(
                    (
                        comment_id,
                        metadata_labels[comment_id],
                        official_labels[comment_id]
                    )
                )

        total_mismatches += len(mismatches)

        print_section(split.upper())

        print("Official TSV records:", len(official_ids))
        print("Unique metadata IDs:", len(metadata_ids))
        print("Common IDs:", len(common_ids))
        print("Missing from metadata:", len(missing_from_metadata))
        print("Missing from official TSV:", len(missing_from_official))
        print("Label mismatches:", len(mismatches))

        for comment_id, metadata_set, official_set in mismatches[
            :SAMPLE_MISMATCH_COUNT
        ]:

            print("\nMismatch:", comment_id)
            print("Metadata labels:", sorted(metadata_set))
            print("Official labels:", sorted(official_set))

    print_section("Overall verification")

    print("Total label mismatches:", total_mismatches)

    print(
        "Note: metadata annotations and official split labels "
        "are not necessarily identical."
    )


# ============================================================
# 10. OFFICIAL LABEL DISTRIBUTION
# ============================================================

def audit_official_label_distribution(archive):

    print_header("10. OFFICIAL LABEL DISTRIBUTION")

    overall_counts = Counter()
    overall_combinations = Counter()
    overall_records = 0
    overall_multilabel = 0

    for split, filename in OFFICIAL_SPLITS.items():

        rows = read_tsv(archive, filename)

        label_counts = Counter()
        combination_counts = Counter()

        total_records = 0
        multilabel_records = 0

        for row in rows:

            total_records += 1
            labels = parse_labels(row["labels"])

            label_counts.update(labels)

            combination = tuple(sorted(labels))
            combination_counts[combination] += 1

            if len(labels) > 1:
                multilabel_records += 1

        overall_records += total_records
        overall_multilabel += multilabel_records
        overall_counts.update(label_counts)
        overall_combinations.update(combination_counts)

        print_section(f"SPLIT: {split.upper()}")

        print("Total records:", total_records)
        print("Multi-label records:", multilabel_records)

        print("\nIndividual label counts:")

        for label, count in label_counts.most_common():

            percentage = count / total_records * 100

            print(
                f"{label:30s} "
                f"{count:6d} "
                f"({percentage:.2f}%)"
            )

        print("\nLabel combinations:")

        for combination, count in combination_counts.most_common():
            print(f"{combination}: {count}")

    print_section("OVERALL OFFICIAL DISTRIBUTION")

    print("Total records:", overall_records)
    print("Multi-label records:", overall_multilabel)

    print("\nOverall label counts:")

    for label, count in overall_counts.most_common():

        percentage = count / overall_records * 100

        print(
            f"{label:30s} "
            f"{count:6d} "
            f"({percentage:.2f}%)"
        )

    print("\nOverall label combinations:")

    for combination, count in overall_combinations.most_common():
        print(f"{combination}: {count}")


# ============================================================
# 11. FINAL AUDIT SUMMARY
# ============================================================

def final_summary(archive, metadata):

    print_header("11. FINAL DATASET AUDIT SUMMARY")

    print("Dataset: Contextual Abuse Dataset (CAD)")
    print("Version: v1.1")
    print("Source archive:", ZIP_PATH.name)

    print("Metadata rows:", len(metadata))

    unique_ids = {
        row["info_id"]
        for row in metadata
    }

    print("Unique metadata comment IDs:", len(unique_ids))

    print_section("Official split sizes")

    total = 0

    for split, filename in OFFICIAL_SPLITS.items():

        rows = read_tsv(archive, filename)

        print(f"{split}: {len(rows)}")

        total += len(rows)

    print("Total official records:", total)

    print_section("Project decisions")

    print("1. Official v1.1 split TSVs are the label source of truth.")
    print("2. Combined metadata is used for context and thread information.")
    print("3. Repeated metadata IDs must not be blindly treated as separate comments.")
    print("4. Multi-label targets must be preserved.")
    print("5. Context reconstruction must respect train/dev/test boundaries.")
    print("6. Missing parent comments must be handled explicitly.")
    print("7. Password-protected datasets are excluded from this audit.")


# ============================================================
# MAIN
# ============================================================

def main():

    print_header("CONTEXT-AWARE TOXICITY CLASSIFIER")
    print("CAD v1.1 COMPLETE DATASET AUDIT")

    audit_dataset_inventory()

    if not ZIP_PATH.exists():

        print("\nERROR: CAD archive not found.")
        print("Expected location:", ZIP_PATH)
        return

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:

        audit_archive(archive)

        required_files = [
            METADATA_FILE,
            *OFFICIAL_SPLITS.values()
        ]

        missing_files = [
            filename
            for filename in required_files
            if filename not in archive.namelist()
        ]

        if missing_files:

            print("\nERROR: Required files missing from archive:")

            for filename in missing_files:
                print(filename)

            return

        audit_tsv_structure(archive)

        metadata = read_tsv(archive, METADATA_FILE)

        audit_duplicate_ids(metadata)

        audit_metadata_labels(metadata)

        audit_context_availability(metadata)

        audit_threads(metadata)

        audit_context_examples(metadata)

        audit_official_label_verification(
            archive,
            metadata
        )

        audit_official_label_distribution(archive)

        final_summary(archive, metadata)

    print_header("ALL DATASET AUDITS COMPLETED")


if __name__ == "__main__":
    main()
