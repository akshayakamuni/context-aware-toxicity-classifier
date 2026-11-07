
from pathlib import Path
from collections import defaultdict, Counter
import zipfile
import csv


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ZIP_PATH = PROJECT_ROOT / "datasets" / "raw" / "data.zip"

METADATA_FILE = "data/cad_v1_1.tsv"

OFFICIAL_SPLITS = {
    "train": "data/cad_v1_1_train.tsv",
    "dev": "data/cad_v1_1_dev.tsv",
    "test": "data/cad_v1_1_test.tsv"
}

OFFICIAL_SPLIT_NAMES = set(OFFICIAL_SPLITS)

SAMPLE_LIMIT = 5


# ============================================================
# HELPERS
# ============================================================

def read_tsv(archive, filename):

    with archive.open(filename) as file:

        reader = csv.DictReader(
            (line.decode("utf-8-sig") for line in file),
            delimiter="\t"
        )

        return list(reader)


def clean(value):

    if value is None:
        return None

    value = value.strip()

    if value in ("", "NA", "None"):
        return None

    return value


def print_header(title):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def print_section(title):

    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


# ============================================================
# LOAD AND INDEX METADATA
# ============================================================

def build_metadata_index(metadata):

    metadata_by_id = defaultdict(list)

    for row in metadata:

        comment_id = clean(row["info_id"])

        if comment_id:
            metadata_by_id[comment_id].append(row)

    return metadata_by_id


# ============================================================
# AUDIT METADATA CONSISTENCY
# ============================================================

def audit_metadata_consistency(metadata_by_id):

    print_header("1. METADATA CONSISTENCY")

    duplicate_ids = {
        comment_id: rows
        for comment_id, rows in metadata_by_id.items()
        if len(rows) > 1
    }

    conflicting_parent = []
    conflicting_text = []
    conflicting_split = []

    for comment_id, rows in duplicate_ids.items():

        parents = {
            clean(row["info_id.parent"])
            for row in rows
        }

        texts = {
            clean(row["meta_text"])
            for row in rows
        }

        splits = {
            clean(row["split"])
            for row in rows
        }

        if len(parents) > 1:
            conflicting_parent.append(comment_id)

        if len(texts) > 1:
            conflicting_text.append(comment_id)

        if len(splits) > 1:
            conflicting_split.append(comment_id)

    print("Unique metadata IDs:", len(metadata_by_id))
    print("Repeated IDs:", len(duplicate_ids))
    print("IDs with conflicting parent IDs:", len(conflicting_parent))
    print("IDs with conflicting text:", len(conflicting_text))
    print("IDs with conflicting split assignments:", len(conflicting_split))

    if conflicting_parent:
        print("Sample conflicting parent IDs:", conflicting_parent[:SAMPLE_LIMIT])

    if conflicting_text:
        print("Sample conflicting text IDs:", conflicting_text[:SAMPLE_LIMIT])

    if conflicting_split:
        print("Sample conflicting split IDs:", conflicting_split[:SAMPLE_LIMIT])


# ============================================================
# BUILD OFFICIAL SPLIT INDEX
# ============================================================

def build_official_split_index(archive):

    official_records = {}
    official_split_by_id = {}

    print_header("2. OFFICIAL SPLIT INDEX")

    for split, filename in OFFICIAL_SPLITS.items():

        rows = read_tsv(archive, filename)

        official_records[split] = {
            row["id"]: row
            for row in rows
        }

        for row in rows:
            official_split_by_id[row["id"]] = split

        print(f"{split}: {len(rows)} records")

    print("Total official records:", len(official_split_by_id))

    return official_records, official_split_by_id


# ============================================================
# CONTEXT RECONSTRUCTION AUDIT
# ============================================================

def audit_context_reconstruction(
    official_records,
    official_split_by_id,
    metadata_by_id
):

    print_header("3. CONTEXT RECONSTRUCTION AUDIT")

    overall_counts = Counter()
    split_results = {}

    samples = defaultdict(list)

    for split, records in official_records.items():

        counts = Counter()

        for child_id, official_row in records.items():

            child_metadata_rows = metadata_by_id.get(child_id)

            if not child_metadata_rows:

                counts["child_metadata_missing"] += 1
                continue

            # Earlier duplicate-ID audit found identical text,
            # parent ID and split for repeated annotation rows.
            child_metadata = child_metadata_rows[0]

            parent_id = clean(child_metadata["info_id.parent"])

            context_annotations = {
                clean(row["annotation_Context"])
                for row in child_metadata_rows
            }

            context_annotations.discard(None)

            if not parent_id:

                counts["no_parent_id"] += 1
                continue

            if parent_id not in metadata_by_id:

                counts["parent_missing_from_metadata"] += 1

                if len(samples["missing_parent"]) < SAMPLE_LIMIT:
                    samples["missing_parent"].append(
                        (split, child_id, parent_id)
                    )

                continue

            parent_rows = metadata_by_id[parent_id]
            parent_metadata = parent_rows[0]

            parent_text = clean(parent_metadata["meta_text"])

            parent_metadata_splits = {
                clean(row["split"])
                for row in parent_rows
            }

            parent_metadata_splits.discard(None)

            # Parent exists, but its text is unavailable.
            if not parent_text:

                counts["parent_text_missing"] += 1
                continue

            parent_official_split = official_split_by_id.get(parent_id)

            # Parent is present in metadata but excluded from
            # the official train/dev/test dataset.
            if parent_official_split is None:

                counts["parent_excluded_from_official_splits"] += 1

                if len(samples["excluded_parent"]) < SAMPLE_LIMIT:
                    samples["excluded_parent"].append(
                        (split, child_id, parent_id,
                         sorted(parent_metadata_splits))
                    )

                continue

            # Parent belongs to the same official split.
            if parent_official_split == split:

                counts["same_split_parent_available"] += 1

                if len(samples["same_split"]) < SAMPLE_LIMIT:
                    samples["same_split"].append(
                        (split, child_id, parent_id)
                    )

            # Parent belongs to a different official split.
            else:

                counts["cross_split_parent"] += 1

                if len(samples["cross_split"]) < SAMPLE_LIMIT:
                    samples["cross_split"].append(
                        (
                            split,
                            child_id,
                            parent_id,
                            parent_official_split
                        )
                    )

            if context_annotations:
                counts["with_context_annotation"] += 1
            else:
                counts["without_context_annotation"] += 1

        split_results[split] = counts
        overall_counts.update(counts)

    # Display split-level results.
    for split, counts in split_results.items():

        print_section(f"{split.upper()} SPLIT")

        print("Official records:", len(official_records[split]))

        for category, count in counts.items():
            print(f"{category}: {count}")

    # Display overall results.
    print_header("4. OVERALL CONTEXT AVAILABILITY")

    total_records = sum(
        len(records)
        for records in official_records.values()
    )

    print("Total official records:", total_records)

    categories = [
        "child_metadata_missing",
        "no_parent_id",
        "parent_missing_from_metadata",
        "parent_text_missing",
        "parent_excluded_from_official_splits",
        "same_split_parent_available",
        "cross_split_parent",
    ]

    for category in categories:
        print(f"{category}: {overall_counts[category]}")

    classified = sum(overall_counts[c] for c in categories)

    print("Records accounted for:", classified)
    print("Unaccounted records:", total_records - classified)

    # Display examples.
    print_header("5. SAMPLE RELATIONSHIPS")

    for category, examples in samples.items():

        print_section(category.upper())

        for example in examples:
            print(example)

    return split_results, overall_counts


# ============================================================
# CONTEXT ANNOTATION ANALYSIS
# ============================================================

def audit_context_annotations(
    official_records,
    metadata_by_id
):

    print_header("6. CONTEXT ANNOTATION ANALYSIS")

    for split, records in official_records.items():

        counts = Counter()

        for comment_id in records:

            rows = metadata_by_id.get(comment_id, [])

            if not rows:
                counts["Missing metadata"] += 1
                continue

            annotations = {
                clean(row["annotation_Context"])
                for row in rows
            }

            annotations.discard(None)

            if not annotations:
                counts["No context annotation"] += 1
            else:
                for annotation in annotations:
                    counts[annotation] += 1

        print_section(split.upper())

        for annotation, count in counts.most_common():
            print(f"{annotation}: {count}")


# ============================================================
# PARENT-CHILD LEAKAGE ANALYSIS
# ============================================================

def audit_cross_split_relationships(
    official_records,
    official_split_by_id,
    metadata_by_id
):

    print_header("7. CROSS-SPLIT RELATIONSHIP ANALYSIS")

    relationships = set()
    cross_split_relationships = []

    for child_split, records in official_records.items():

        for child_id in records:

            rows = metadata_by_id.get(child_id, [])

            if not rows:
                continue

            parent_id = clean(rows[0]["info_id.parent"])

            if not parent_id:
                continue

            parent_split = official_split_by_id.get(parent_id)

            if parent_split is None:
                continue

            relationship = (
                child_id,
                parent_id,
                child_split,
                parent_split
            )

            if relationship in relationships:
                continue

            relationships.add(relationship)

            if child_split != parent_split:
                cross_split_relationships.append(relationship)

    print("Unique official parent-child relationships:", len(relationships))

    print(
        "Cross-split relationships:",
        len(cross_split_relationships)
    )

    print_section("Cross-split direction")

    direction_counts = Counter(
        (child_split, parent_split)
        for _, _, child_split, parent_split
        in cross_split_relationships
    )

    for direction, count in direction_counts.items():
        print(f"Child {direction[0]} -> Parent {direction[1]}: {count}")

    print_section("Sample cross-split relationships")

    for relationship in cross_split_relationships[:10]:
        print(relationship)


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(split_results, overall_counts):

    print_header("8. CONTEXT RECONSTRUCTION SUMMARY")

    print("Context reconstruction audit completed.")

    print_section("Initial findings")

    print(
        "Same-split parents available:",
        overall_counts["same_split_parent_available"]
    )

    print(
        "Cross-split parents:",
        overall_counts["cross_split_parent"]
    )

    print(
        "Parents missing from metadata:",
        overall_counts["parent_missing_from_metadata"]
    )

    print(
        "Parents excluded from official splits:",
        overall_counts["parent_excluded_from_official_splits"]
    )

    print_section("Proposed safety rules")

    print("1. Never use a parent from another official split.")
    print("2. Never use test labels during training.")
    print("3. Do not fabricate missing parent text.")
    print("4. Preserve the original official child labels.")
    print("5. Keep context availability information for evaluation.")
    print("6. Investigate excluded parents separately before using them.")
    print("7. Use only verified same-split parent text in the initial baseline.")


# ============================================================
# MAIN
# ============================================================

def main():

    print_header("CONTEXT-AWARE TOXICITY CLASSIFIER")
    print("LEAKAGE-SAFE CONTEXT RECONSTRUCTION AUDIT")

    if not ZIP_PATH.exists():

        print("ERROR: CAD archive not found:", ZIP_PATH)
        return

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:

        required_files = [
            METADATA_FILE,
            *OFFICIAL_SPLITS.values()
        ]

        missing = [
            filename
            for filename in required_files
            if filename not in archive.namelist()
        ]

        if missing:

            print("ERROR: Missing archive files:")

            for filename in missing:
                print(filename)

            return

        metadata = read_tsv(archive, METADATA_FILE)

        metadata_by_id = build_metadata_index(metadata)

        audit_metadata_consistency(metadata_by_id)

        official_records, official_split_by_id = (
            build_official_split_index(archive)
        )

        split_results, overall_counts = (
            audit_context_reconstruction(
                official_records,
                official_split_by_id,
                metadata_by_id
            )
        )

        audit_context_annotations(
            official_records,
            metadata_by_id
        )

        audit_cross_split_relationships(
            official_records,
            official_split_by_id,
            metadata_by_id
        )

        final_summary(
            split_results,
            overall_counts
        )

    print_header("CONTEXT RECONSTRUCTION AUDIT COMPLETED")


if __name__ == "__main__":
    main()
