"""
extract_dataset.py

Extract features from every ECG record
and save them as a CSV.

Author: ECG Feature Extraction Pipeline
"""

import os
import pandas as pd
from tqdm import tqdm

from load_record import load_record
from feature_extractor import extract_features


# ----------------------------------------------------
# Change these paths
# ----------------------------------------------------

DATASET_PATH = ""

OUTPUT_CSV = "ecg_features.csv"

FAILED_LOG = "failed_records.txt"


# ----------------------------------------------------
# Find every record
# ----------------------------------------------------

records = []

for file in os.listdir(DATASET_PATH):

    if file.endswith(".hea"):

        record_name = os.path.splitext(file)[0]

        records.append(

            os.path.join(
                DATASET_PATH,
                record_name
            )

        )

records.sort()

print(f"Found {len(records)} ECG records")


# ----------------------------------------------------
# Extract Features
# ----------------------------------------------------

dataset = []

failed = []

for record_path in tqdm(records):

    try:

        ecg = load_record(record_path)

        features = extract_features(ecg)

        # Store record name
        features["record_name"] = ecg.record_name

        # ------------------------------------
        # LABEL
        # ------------------------------------
        #
        # Change this according to your dataset
        #

        label = 0

        comments = " ".join(ecg.comments).lower()

        if "congenital" in comments:

            label = 1

        features["label"] = label

        dataset.append(features)

    except Exception as e:

        failed.append(

            f"{record_path} --> {str(e)}"

        )


# ----------------------------------------------------
# Save CSV
# ----------------------------------------------------

df = pd.DataFrame(dataset)

df.to_csv(

    OUTPUT_CSV,

    index=False

)

print()

print("="*40)

print("Extraction Complete")

print(f"Patients : {len(df)}")

print(f"Failed   : {len(failed)}")

print(f"CSV      : {OUTPUT_CSV}")

print("="*40)


# ----------------------------------------------------
# Save Failed Records
# ----------------------------------------------------

if len(failed):

    with open(FAILED_LOG, "w") as f:

        for record in failed:

            f.write(record + "\n")

    print(f"Failed log saved to {FAILED_LOG}")