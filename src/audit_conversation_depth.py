
from pathlib import Path
from collections import Counter, defaultdict
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

EXPECTED_OFFICIAL_RECORDS = 23417

SAMPLE_LIMIT = 5
MAX_CHAIN_DEPTH = 20


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
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_section(title):
    print("\n" + "-" * 65)
    print(title)
    print("-" * 65)


# ============================================================
# BUILD DATASET INDEXES
# ============================================================

def build_indexes(archive):

    metadata = read_tsv(archive, METADATA_FILE)

    metadata_by_id = defaultdict(list)

    for row in metadata:
        comment_id = clean(row["info_id"])

        if comment_id:
            metadata_by_id[comment_id].append(row)

    # Official split files are the source of truth.
    official_split_by_id = {}

    for split, filename in OFFICIAL_SPLITS.items():

        rows = read_tsv(archive, filename)

        for row in rows:

            comment_id = clean(row["id"])

            if comment_id in official_split_by_id:
                raise ValueError(
                    f"Duplicate official comment ID: {comment_id}"
                )

            official_split_by_id[comment_id] = split

    # One representative metadata row per unique comment ID.
    # Earlier audits established that duplicate rows have
    # consistent text, parent ID and split assignments.
    metadata_index = {}

    for comment_id, rows in metadata_by_id.items():
        metadata_index[comment_id] = rows[0]

    # Preserve all distinct context annotations for each ID.
    # This prevents duplicate rows from increasing record counts.
    context_annotations_by_id = {}

    for comment_id, rows in metadata_by_id.items():

        annotations = set()

        for row in rows:
            annotation = clean(row.get("annotation_Context"))

            if annotation:
                annotations.add(annotation)

        context_annotations_by_id[comment_id] = annotations

    return (
        metadata,
        metadata_index,
        official_split_by_id,
        context_annotations_by_id
    )


# ============================================================
# RECONSTRUCT ONE CONVERSATION CHAIN
# ============================================================

def reconstruct_chain(
    child_id,
    child_split,
    metadata_index,
    official_split_by_id
):

    chain = []

    visited = {child_id}

    current_id = child_id

    termination_reason = "maximum_depth_reached"

    for depth in range(1, MAX_CHAIN_DEPTH + 1):

        current = metadata_index.get(current_id)

        if current is None:
            termination_reason = "current_record_missing"
            break

        parent_id = clean(current["info_id.parent"])

        if parent_id is None:
            termination_reason = "root_or_no_parent"
            break

        if parent_id in visited:
            termination_reason = "cycle_detected"
            break

        visited.add(parent_id)

        parent = metadata_index.get(parent_id)

        if parent is None:
            termination_reason = "parent_missing_from_metadata"
            break

        parent_text = clean(parent["meta_text"])

        if not parent_text:
            termination_reason = "parent_text_missing"
            break

        parent_split = official_split_by_id.get(parent_id)

        if parent_split is None:
            termination_reason = "parent_excluded_from_official_splits"
            break

        if parent_split != child_split:
            termination_reason = "cross_split_parent"
            break

        chain.append({
            "id": parent_id,
            "text": parent_text,
            "split": parent_split,
            "depth": depth,
            "thread_id": clean(parent["info_thread.id"]),
            "order": clean(parent["info_order"])
        })

        current_id = parent_id

    else:
        termination_reason = "maximum_depth_reached"

    return chain, termination_reason


# ============================================================
# AUDIT CONVERSATION DEPTH
# ============================================================

def audit_conversation_depth(
    metadata_index,
    official_split_by_id
):

    print_header("1. CONVERSATION DEPTH ANALYSIS")

    depth_distribution = Counter()
    termination_counts = Counter()

    split_depths = defaultdict(Counter)

    chain_samples = defaultdict(list)

    total_records = 0
    total_available_ancestors = 0

    # IMPORTANT:
    # Iterate over unique official IDs, not metadata rows.
    for child_id, child_split in official_split_by_id.items():

        total_records += 1

        chain, termination = reconstruct_chain(
            child_id,
            child_split,
            metadata_index,
            official_split_by_id
        )

        depth = len(chain)

        depth_distribution[depth] += 1
        split_depths[child_split][depth] += 1

        termination_counts[termination] += 1

        total_available_ancestors += depth

        if len(chain_samples[depth]) < SAMPLE_LIMIT:

            child_row = metadata_index.get(child_id)

            chain_samples[depth].append({
                "child_id": child_id,
                "split": child_split,
                "child_text": (
                    clean(child_row["meta_text"])
                    if child_row else None
                ),
                "ancestors": chain,
                "termination": termination
            })

    print("Official target records:", total_records)

    print(
        "Total reconstructable ancestor relationships:",
        total_available_ancestors
    )

    print(
        "Unique official IDs missing from metadata:",
        sum(
            1 for comment_id in official_split_by_id
            if comment_id not in metadata_index
        )
    )

    print_section("Conversation depth distribution")

    for depth in sorted(depth_distribution):

        count = depth_distribution[depth]

        percentage = count / total_records * 100

        print(
            f"{depth} available ancestors: "
            f"{count} records ({percentage:.2f}%)"
        )

    print_section("Split-wise conversation depth")

    for split in ("train", "dev", "test"):

        print(f"\n{split.upper()}")

        for depth in sorted(split_depths[split]):

            print(
                f"Depth {depth}: "
                f"{split_depths[split][depth]}"
            )

    print_section("Conversation termination reasons")

    for reason, count in termination_counts.most_common():

        percentage = count / total_records * 100

        print(
            f"{reason}: {count} "
            f"({percentage:.2f}%)"
        )

    print_section("Average reconstructable ancestors")

    average = total_available_ancestors / total_records

    print(
        f"Average ancestors per official record: {average:.4f}"
    )

    print_section("Sample conversation chains")

    for depth in sorted(chain_samples):

        print(f"\nDEPTH: {depth}")

        for sample in chain_samples[depth][:2]:

            print("\nTarget:", sample["child_id"])
            print("Split:", sample["split"])

            if sample["child_text"]:
                print("Target text:", sample["child_text"][:200])

            for ancestor in sample["ancestors"]:

                print(
                    f"  Level {ancestor['depth']}: "
                    f"{ancestor['id']}"
                )

                print(
                    "  Text:",
                    ancestor["text"][:200]
                )

            print("Termination:", sample["termination"])

    # Every unique official record must appear exactly once.
    counted_records = sum(depth_distribution.values())

    if counted_records != total_records:
        raise AssertionError(
            "Depth distribution count does not match "
            "the number of official records."
        )

    return depth_distribution, termination_counts


# ============================================================
# ANNOTATION AND CONTEXT AVAILABILITY COMPARISON
# ============================================================

def audit_annotation_vs_context(
    metadata_index,
    official_split_by_id,
    context_annotations_by_id
):

    print_header("2. CONTEXT ANNOTATION VS ACTUAL AVAILABILITY")

    comparison = defaultdict(Counter)

    # One comparison per unique official comment ID.
    for child_id, split in official_split_by_id.items():

        chain, _ = reconstruct_chain(
            child_id,
            split,
            metadata_index,
            official_split_by_id
        )

        if chain:
            availability = "Parent context available"
        else:
            availability = "Parent context unavailable"

        annotations = context_annotations_by_id.get(
            child_id, set()
        )

        if not annotations:
            annotation_key = "No context annotation"
        else:
            annotation_key = " + ".join(sorted(annotations))

        comparison[annotation_key][availability] += 1

    print_section("Annotation and availability comparison")

    for annotation, counts in sorted(comparison.items()):

        print(f"\n{annotation}")

        for availability, count in sorted(counts.items()):

            print(f"{availability}: {count}")

    total_compared = sum(
        sum(counts.values())
        for counts in comparison.values()
    )

    print("\nTotal unique official records compared:", total_compared)

    if total_compared != len(official_split_by_id):
        raise AssertionError(
            "Annotation comparison count does not match "
            "the official record count."
        )


# ============================================================
# AUDIT THREAD CONSISTENCY
# ============================================================

def audit_thread_consistency(metadata_index):

    print_header("3. THREAD CONSISTENCY")

    thread_members = defaultdict(list)

    for comment_id, row in metadata_index.items():

        thread_id = clean(row["info_thread.id"])

        if thread_id:
            thread_members[thread_id].append(comment_id)

    total_threads = len(thread_members)

    parent_thread_mismatches = 0
    missing_parent_thread = 0
    checked_relationships = 0

    examples = []

    for child_id, row in metadata_index.items():

        parent_id = clean(row["info_id.parent"])

        if not parent_id:
            continue

        parent = metadata_index.get(parent_id)

        if parent is None:
            continue

        child_thread = clean(row["info_thread.id"])
        parent_thread = clean(parent["info_thread.id"])

        checked_relationships += 1

        if child_thread is None or parent_thread is None:

            missing_parent_thread += 1
            continue

        if child_thread != parent_thread:

            parent_thread_mismatches += 1

            if len(examples) < SAMPLE_LIMIT:

                examples.append(
                    (
                        child_id,
                        parent_id,
                        child_thread,
                        parent_thread
                    )
                )

    print("Unique threads:", total_threads)

    print(
        "Parent-child relationships checked:",
        checked_relationships
    )

    print(
        "Relationships with missing thread ID:",
        missing_parent_thread
    )

    print(
        "Parent-child thread ID mismatches:",
        parent_thread_mismatches
    )

    if examples:

        print_section("Thread mismatch examples")

        for example in examples:
            print(example)


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(depth_distribution, termination_counts):

    print_header("4. CONVERSATION DEPTH AUDIT SUMMARY")

    print("Depth distribution:")

    for depth, count in sorted(depth_distribution.items()):
        print(f"{depth} ancestors: {count}")

    print("\nTermination reasons:")

    for reason, count in termination_counts.most_common():
        print(f"{reason}: {count}")

    print_section("Research questions to answer")

    print("1. Is immediate parent context sufficient?")
    print("2. Does additional conversation history add information?")
    print("3. How much context is lost because of missing ancestors?")
    print("4. Do human context annotations align with available history?")
    print("5. Should the model use one parent or multiple ancestors?")

    print_section("Important safeguards")

    print("1. Use only same-split parent text.")
    print("2. Do not use parent labels as model inputs.")
    print("3. Do not fabricate missing ancestors.")
    print("4. Detect cycles before reconstructing conversation chains.")
    print("5. Preserve official child labels.")
    print("6. Keep the original train/dev/test boundaries.")
    print("7. Count each official comment ID exactly once.")
    print("8. Keep missing-context records in the dataset.")


# ============================================================
# MAIN
# ============================================================

def main():

    print_header("CONTEXT-AWARE TOXICITY CLASSIFIER")

    print("CONVERSATION DEPTH AND HISTORY AUDIT")

    if not ZIP_PATH.exists():

        print("\nERROR: CAD archive not found.")
        print(ZIP_PATH)
        return

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:

        required_files = [
            METADATA_FILE,
            *OFFICIAL_SPLITS.values()
        ]

        archive_files = set(archive.namelist())

        missing = [
            filename
            for filename in required_files
            if filename not in archive_files
        ]

        if missing:

            print("\nERROR: Missing archive files:")

            for filename in missing:
                print(filename)

            return

        (
            metadata,
            metadata_index,
            official_split_by_id,
            context_annotations_by_id
        ) = build_indexes(archive)

        print_section("Dataset counting validation")

        print("Raw metadata rows:", len(metadata))
        print("Unique metadata IDs:", len(metadata_index))
        print("Unique official IDs:", len(official_split_by_id))

        split_counts = Counter(official_split_by_id.values())

        for split in ("train", "dev", "test"):
            print(
                f"Official {split} records:",
                split_counts[split]
            )

        if len(official_split_by_id) != EXPECTED_OFFICIAL_RECORDS:

            raise ValueError(
                f"Expected {EXPECTED_OFFICIAL_RECORDS} official records, "
                f"but found {len(official_split_by_id)}."
            )

        print("\nOfficial record count validation: PASSED")

        depth_distribution, termination_counts = (
            audit_conversation_depth(
                metadata_index,
                official_split_by_id
            )
        )

        audit_annotation_vs_context(
            metadata_index,
            official_split_by_id,
            context_annotations_by_id
        )

        audit_thread_consistency(metadata_index)

        final_summary(
            depth_distribution,
            termination_counts
        )

    print_header("CONVERSATION DEPTH AUDIT COMPLETED")


if __name__ == "__main__":
    main()
