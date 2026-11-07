
import zipfile
from pathlib import Path
from collections import Counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PATH = PROJECT_ROOT / "datasets" / "raw" / "data.zip"

files = [
    "cad_v1_1_train.tsv",
    "cad_v1_1_dev.tsv",
    "cad_v1_1_test.tsv"
]

label_values = Counter()

with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:

    for filename in files:

        matches = [
            name for name in archive.namelist()
            if Path(name).name == filename
        ]

        if len(matches) != 1:
            raise ValueError(f"Could not uniquely locate {filename}")

        with archive.open(matches[0]) as file:

            df = pd.read_csv(
                file,
                sep="\t",
                dtype=str,
                keep_default_na=False
            )

        label_values.update(df["labels"])

print("\n========== DISTINCT LABEL STRINGS ==========")

for label, count in label_values.most_common():

    print(repr(label), ":", count)

print("\nTotal distinct label combinations:", len(label_values))
print("Total records:", sum(label_values.values()))
