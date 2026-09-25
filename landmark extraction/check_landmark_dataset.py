import os
import csv
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

# This must point to your Landmark_Data folder.
LANDMARK_FOLDER = r"Landmark_Data"

# ============================================================
# QUALITY CHECK
# ============================================================

print()
print("================================================")
print("       LANDMARK DATASET QUALITY CHECK")
print("================================================")
print()

if not os.path.isdir(LANDMARK_FOLDER):

    print("ERROR: Landmark_Data folder not found:")
    print(LANDMARK_FOLDER)
    print()
    print("Make sure this script is inside the main project folder.")
    exit()


# ------------------------------------------------------------
# Find every .npy file
# ------------------------------------------------------------

npy_files = []

for current_folder, subfolders, files in os.walk(LANDMARK_FOLDER):

    for file in files:

        if file.lower().endswith(".npy"):

            npy_files.append(
                os.path.join(current_folder, file)
            )

npy_files.sort()

print("Total .npy files found:", len(npy_files))
print()


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

valid_files = 0
wrong_shape = []
wrong_dtype = []
nan_files = []
inf_files = []

frame_counts = []

# Hand information is inferred from the zero-filled landmark blocks.
# Pose = columns 0:99
# Left hand = columns 99:162
# Right hand = columns 162:225

low_left = []
low_right = []
low_either = []

# ------------------------------------------------------------
# Check every file
# ------------------------------------------------------------

for number, file_path in enumerate(npy_files, start=1):

    try:

        data = np.load(
            file_path,
            mmap_mode="r"
        )

        # ----------------------------------------------------
        # Shape
        # ----------------------------------------------------

        if data.ndim != 2 or data.shape[1] != 225:

            wrong_shape.append(
                (
                    file_path,
                    data.shape
                )
            )

            continue

        # ----------------------------------------------------
        # Data type
        # ----------------------------------------------------

        if data.dtype != np.float32:

            wrong_dtype.append(
                (
                    file_path,
                    str(data.dtype)
                )
            )

        # ----------------------------------------------------
        # NaN / Infinity
        # ----------------------------------------------------

        if np.isnan(data).any():

            nan_files.append(file_path)

        if np.isinf(data).any():

            inf_files.append(file_path)

        # ----------------------------------------------------
        # Frame count
        # ----------------------------------------------------

        frame_counts.append(data.shape[0])

        # ----------------------------------------------------
        # Hand detection estimate
        #
        # A frame is considered detected if the corresponding
        # 63 landmark values are not all zero.
        # ----------------------------------------------------

        left = np.asarray(data[:, 99:162])
        right = np.asarray(data[:, 162:225])

        left_detected = np.any(left != 0, axis=1)
        right_detected = np.any(right != 0, axis=1)

        left_count = int(np.sum(left_detected))
        right_count = int(np.sum(right_detected))

        total_frames = data.shape[0]

        left_ratio = left_count / total_frames
        right_ratio = right_count / total_frames

        # Flag only very low detection.
        # This does NOT delete anything.
        if left_ratio < 0.20:

            low_left.append(
                (
                    file_path,
                    left_count,
                    total_frames,
                    left_ratio
                )
            )

        if right_ratio < 0.20:

            low_right.append(
                (
                    file_path,
                    right_count,
                    total_frames,
                    right_ratio
                )
            )

        if left_ratio < 0.20 or right_ratio < 0.20:

            low_either.append(
                (
                    file_path,
                    left_count,
                    right_count,
                    total_frames,
                    left_ratio,
                    right_ratio
                )
            )

        valid_files += 1

    except Exception as e:

        print()
        print("ERROR reading:", file_path)
        print("Reason:", e)
        print()


# ============================================================
# RESULTS
# ============================================================

print("================================================")
print("                 RESULTS")
print("================================================")
print()

print("Total .npy files:", len(npy_files))
print("Valid 225-feature files:", valid_files)
print("Wrong shape:", len(wrong_shape))
print("Wrong data type:", len(wrong_dtype))
print("NaN files:", len(nan_files))
print("Infinite-value files:", len(inf_files))

print()

if frame_counts:

    print("Frame count:")
    print("Minimum:", min(frame_counts))
    print("Maximum:", max(frame_counts))
    print("Average:", round(sum(frame_counts) / len(frame_counts), 2))

print()
print("------------------------------------------------")
print("LOW HAND-DETECTION SAMPLES")
print("Threshold: below 20% detected frames")
print("------------------------------------------------")
print()

print("Low left-hand samples:", len(low_left))
print("Low right-hand samples:", len(low_right))
print("Low either-hand samples:", len(low_either))

print()


# ------------------------------------------------------------
# Print suspicious files
# ------------------------------------------------------------

if low_either:

    print("Files flagged for inspection:")
    print()

    for item in low_either:

        file_path = item[0]
        left_count = item[1]
        right_count = item[2]
        total_frames = item[3]
        left_ratio = item[4]
        right_ratio = item[5]

        print(
            os.path.relpath(
                file_path,
                LANDMARK_FOLDER
            )
        )

        print(
            "  Left :",
            left_count,
            "/",
            total_frames,
            "(",
            round(left_ratio * 100, 1),
            "%)"
        )

        print(
            "  Right:",
            right_count,
            "/",
            total_frames,
            "(",
            round(right_ratio * 100, 1),
            "%)"
        )

        print()


# ------------------------------------------------------------
# Final interpretation
# ------------------------------------------------------------

print("================================================")
print("                 INTERPRETATION")
print("================================================")
print()

if (
    len(wrong_shape) == 0
    and len(nan_files) == 0
    and len(inf_files) == 0
):

    print("STRUCTURE CHECK: PASS")
    print()
    print("All checked files have valid 225-feature structure")
    print("and no NaN/Infinity values were found.")

else:

    print("STRUCTURE CHECK: NEEDS ATTENTION")
    print()
    print("There are structural or numerical issues to inspect.")

print()

print(
    "IMPORTANT: Low hand-detection files are only FLAGGED."
)
print(
    "Nothing is deleted or modified by this script."
)

print()
print("================================================")
print("             QUALITY CHECK FINISHED")
print("================================================")
