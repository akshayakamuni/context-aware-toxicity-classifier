"""
Calculate exact BERT truncation rates for each official label.

Uses the record-level token-length report.
Does not modify the dataset.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "bert_token_lengths_record_level.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "bert_truncation_by_label.csv"
)

MAX_LENGTHS = [64, 128, 256, 512]


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Record-level report not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False
    )

    required_columns = {
        "labels",
        "current_token_length",
        "parent_token_length",
        "pair_token_length"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    token_columns = [
        "current_token_length",
        "parent_token_length",
        "pair_token_length"
    ]

    for column in token_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    rows = []

    for _, record in df.iterrows():

        labels = [
            label.strip()
            for label in str(record["labels"]).split(",")
            if label.strip()
        ]

        for label in labels:

            for input_name, column in [
                ("Current comment", "current_token_length"),
                ("Parent comment", "parent_token_length"),
                ("Current + parent pair", "pair_token_length")
            ]:

                token_length = record[column]

                if pd.isna(token_length):
                    continue

                for limit in MAX_LENGTHS:

                    rows.append({
                        "label": label,
                        "input_type": input_name,
                        "max_length": limit,
                        "token_length": int(token_length),
                        "truncated": token_length > limit
                    })

    detailed = pd.DataFrame(rows)

    summary = (
        detailed
        .groupby(
            ["label", "input_type", "max_length"]
        )
        .agg(
            records=("truncated", "size"),
            truncated_records=("truncated", "sum")
        )
        .reset_index()
    )

    summary["truncation_percentage"] = (
        summary["truncated_records"]
        / summary["records"]
        * 100
    ).round(2)

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 65)
    print("LABEL-WISE BERT TRUNCATION ANALYSIS")
    print("=" * 65)

    print(summary.to_string(index=False))

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()