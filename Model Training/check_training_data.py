import os
import numpy as np
import pandas as pd

CSV_PATH = r"Landmark_Data\landmark_labels.csv"
BASE_FOLDER = r"Landmark_Data"

print("=" * 60)
print("LSTM TRAINING DATA CHECK")
print("=" * 60)

# Load CSV
df = pd.read_csv(CSV_PATH)

print("\nCSV loaded successfully!")
print("Total samples:", len(df))
print("Total categories:", df["Category"].nunique())
print("Total sign classes:", df["Class"].nunique())

print("\nExpected feature count:")
print(df["Feature_Count"].unique())

# Check missing files
missing_files = []
wrong_shape = []
wrong_dtype = []
nan_files = []
inf_files = []

print("\nChecking .npy files...")

for i, row in df.iterrows():

    file_path = row["NPY_Path"].replace("/", os.sep)

    if not os.path.exists(file_path):
        missing_files.append(file_path)
        continue

    try:
        data = np.load(file_path)

        # Shape check
        if len(data.shape) != 2 or data.shape[1] != 225:
            wrong_shape.append((file_path, data.shape))

        # Data type check
        if data.dtype != np.float32:
            wrong_dtype.append((file_path, data.dtype))

        # NaN check
        if np.isnan(data).any():
            nan_files.append(file_path)

        # Infinite value check
        if np.isinf(data).any():
            inf_files.append(file_path)

    except Exception as e:
        print("ERROR:", file_path)
        print(e)

    if (i + 1) % 500 == 0:
        print("Checked:", i + 1, "/", len(df))


print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)

print("\nTotal samples:", len(df))
print("Missing files:", len(missing_files))
print("Wrong shape:", len(wrong_shape))
print("Wrong data type:", len(wrong_dtype))
print("NaN files:", len(nan_files))
print("Infinite-value files:", len(inf_files))

print("\nFrame count statistics:")
print("Minimum:", df["Frame_Count"].min())
print("Maximum:", df["Frame_Count"].max())
print("Average:", round(df["Frame_Count"].mean(), 2))

print("\nSamples per category:")
print(df["Category"].value_counts())

print("\nNumber of sign classes:", df["Class"].nunique())

if len(missing_files) == 0 and len(wrong_shape) == 0 and len(wrong_dtype) == 0 and len(nan_files) == 0 and len(inf_files) == 0:
    print("\nDATASET CHECK PASSED!")
    print("The dataset is ready for LSTM training.")
else:
    print("\nWARNING: Some problems were found.")
    print("Fix them before training.")