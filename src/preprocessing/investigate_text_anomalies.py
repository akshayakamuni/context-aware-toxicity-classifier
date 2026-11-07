
# ============================================================
# CONTEXT-AWARE TOXICITY CLASSIFIER
# STEP 2.2: DETAILED TEXT ANOMALY INVESTIGATION
#
# Dataset: CAD v1.1
# Purpose:
# 1. Inspect empty comments
# 2. Investigate cross-split duplicate texts
# 3. Inspect extremely long comments
# 4. Check duplicate texts with different labels
#
# Original dataset remains unchanged.
# ============================================================

import zipfile
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHIVE_PATH = PROJECT_ROOT / "datasets" / "raw" / "data.zip"

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILES = {
    "train": "cad_v1_1_train.tsv",
    "dev": "cad_v1_1_dev.tsv",
    "test": "cad_v1_1_test.tsv"
}


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    datasets = []

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:

        for split, filename in SPLIT_FILES.items():

            matches = [
                name for name in archive.namelist()
                if Path(name).name == filename
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Could not uniquely locate {filename}"
                )

            with archive.open(matches[0]) as file:

                df = pd.read_csv(
                    file,
                    sep="\t",
                    dtype=str,
                    keep_default_na=False,
                    encoding="utf-8-sig"
                )

            df["split"] = split

            df["text_length"] = df["text"].str.len()

            datasets.append(df)

    return pd.concat(datasets, ignore_index=True)


# ============================================================
# 3. INVESTIGATE EMPTY COMMENTS
# ============================================================

def investigate_empty_comments(df):

    print("\n========== EMPTY COMMENTS ==========")

    empty = df[df["text"].str.strip() == ""][
        ["id", "split", "labels", "text"]
    ].copy()

    print("Number of empty comments:", len(empty))

    if not empty.empty:

        print(empty.to_string(index=False))

    empty.to_csv(
        REPORT_DIR / "empty_comments.csv",
        index=False
    )

    return empty


# ============================================================
# 4. INVESTIGATE CROSS-SPLIT DUPLICATES
# ============================================================

def investigate_cross_split_duplicates(df):

    print("\n========== CROSS-SPLIT DUPLICATES ==========")

    grouped = (
        df.groupby("text", dropna=False)
        .agg(
            record_count=("id", "size"),
            unique_splits=("split", lambda x: sorted(set(x))),
            unique_labels=("labels", lambda x: sorted(set(x))),
            record_ids=("id", list),
            label_values=("labels", list)
        )
        .reset_index()
    )

    duplicates = grouped[
        grouped["unique_splits"].apply(len) > 1
    ].copy()

    duplicates["splits"] = duplicates[
        "unique_splits"
    ].apply(lambda x: ", ".join(x))

    duplicates["text_length"] = duplicates["text"].str.len()

    duplicates["different_label_values"] = duplicates[
        "unique_labels"
    ].apply(lambda x: len(x) > 1)

    duplicates["text_category"] = duplicates[
        "text_length"
    ].apply(
        lambda length:
        "Very short" if length < 10
        else "Short" if length < 50
        else "Medium" if length < 200
        else "Long"
    )

    output = duplicates[
        [
            "text",
            "text_length",
            "text_category",
            "record_count",
            "splits",
            "unique_labels",
            "different_label_values",
            "record_ids",
            "label_values"
        ]
    ].copy()

    output.to_csv(
        REPORT_DIR / "detailed_cross_split_duplicates.csv",
        index=False
    )

    print("Unique cross-split duplicate texts:", len(output))

    print("\nLength category distribution:")

    print(
        output["text_category"].value_counts().to_string()
    )

    print("\nDuplicates with different label strings:")

    different = output[
        output["different_label_values"]
    ]

    print(len(different))

    print("\nSample duplicate comments:")

    for _, row in output.head(15).iterrows():

        print("\nText:", repr(row["text"][:200]))
        print("Length:", row["text_length"])
        print("Splits:", row["splits"])
        print("Labels:", row["unique_labels"])

    return output


# ============================================================
# 5. INVESTIGATE EXTREMELY LONG COMMENTS
# ============================================================

def investigate_long_comments(df):

    print("\n========== LONG COMMENTS ==========")

    long_comments = df[
        df["text_length"] > 1000
    ].copy()

    long_comments = long_comments.sort_values(
        "text_length",
        ascending=False
    )

    output = long_comments[
        [
            "id",
            "split",
            "labels",
            "text_length",
            "text"
        ]
    ].copy()

    output.to_csv(
        REPORT_DIR / "long_comments.csv",
        index=False
    )

    print("Comments above 1000 characters:", len(output))

    print("\nLong comments by split:")

    print(
        output["split"].value_counts().to_string()
    )

    print("\nLong comments by label string:")

    print(
        output["labels"].value_counts().head(20).to_string()
    )

    print("\nFive longest comments:")

    for _, row in output.head(5).iterrows():

        print("\nID:", row["id"])
        print("Split:", row["split"])
        print("Labels:", row["labels"])
        print("Length:", row["text_length"])
        print("Text:", row["text"][:500])

    return output


# ============================================================
# 6. SUMMARY OF ANOMALIES
# ============================================================

def generate_summary(df, empty, duplicates, long_comments):

    summary = pd.DataFrame([

        {
            "inspection": "Total official records",
            "count": len(df)
        },

        {
            "inspection": "Empty comments",
            "count": len(empty)
        },

        {
            "inspection": "Cross-split duplicate texts",
            "count": len(duplicates)
        },

        {
            "inspection": "Cross-split duplicates with different labels",
            "count": int(
                duplicates["different_label_values"].sum()
            )
        },

        {
            "inspection": "Comments above 1000 characters",
            "count": len(long_comments)
        }

    ])

    summary.to_csv(
        REPORT_DIR / "text_anomaly_summary.csv",
        index=False
    )

    print("\n========== ANOMALY SUMMARY ==========")

    print(summary.to_string(index=False))


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("CAD v1.1 - DETAILED TEXT ANOMALY INVESTIGATION")
    print("=" * 65)

    df = load_data()

    print("\nTotal official records:", len(df))

    empty = investigate_empty_comments(df)

    duplicates = investigate_cross_split_duplicates(df)

    long_comments = investigate_long_comments(df)

    generate_summary(
        df,
        empty,
        duplicates,
        long_comments
    )

    print("\n" + "=" * 65)
    print("ANOMALY INVESTIGATION COMPLETED")
    print("=" * 65)

    print("\nReports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
