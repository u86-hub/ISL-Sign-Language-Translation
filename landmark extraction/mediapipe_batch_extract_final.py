import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# ============================================================
# SETTINGS
# ============================================================

# IMPORTANT:
# This must be the ROOT folder of your ALREADY-PREPROCESSED
# dataset.
#
# Example:
# main project/
#     Dataset/
#         Greetings/
#             Hello/
#                 Video_001/
#                     frame_0000.jpg
#                     frame_0001.jpg
#                 Video_002/
#                     ...
#
DATASET_FOLDER = r"Dataset"

# All landmark .npy files + CSV will be placed here.
OUTPUT_FOLDER = r"Landmark_Data"

# ------------------------------------------------------------
# FIRST RUN:
# Process only 5 video/frame folders.
#
# After checking the result, change this to:
#
# MAX_VIDEOS = None
#
# to process the entire dataset.
# ------------------------------------------------------------
MAX_VIDEOS = None

# ------------------------------------------------------------
# KEEP THIS FALSE.
#
# It is safer to keep the original JPG frames until you have
# verified the .npy files.
#
# We can enable deletion later if storage becomes a problem.
# ------------------------------------------------------------
DELETE_FRAMES_AFTER_SUCCESS = False


# ============================================================
# FIND ALL VIDEO FRAME FOLDERS
# ============================================================

def find_frame_folders(root_folder):

    folders = []

    for current_folder, subfolders, files in os.walk(root_folder):

        # A folder containing JPG/PNG files is treated as one
        # preprocessed video folder.
        has_frames = False

        for file in files:

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                has_frames = True
                break

        if has_frames:
            folders.append(current_folder)

    # Make processing order predictable
    folders.sort()

    return folders


# ============================================================
# PROCESS ONE VIDEO FOLDER
# ============================================================

def process_video(frames_folder, output_file, holistic):

    frame_files = []

    for file in os.listdir(frames_folder):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            frame_files.append(file)

    # frame_0000, frame_0001, frame_0002...
    frame_files.sort()

    sequence = []

    pose_detected = 0
    left_hand_detected = 0
    right_hand_detected = 0

    for frame_file in frame_files:

        frame_path = os.path.join(
            frames_folder,
            frame_file
        )

        frame = cv2.imread(frame_path)

        if frame is None:

            print(
                "WARNING: Could not read:",
                frame_path
            )

            continue

        # ----------------------------------------------------
        # OpenCV reads BGR.
        # MediaPipe expects RGB.
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = holistic.process(rgb_frame)

        # ----------------------------------------------------
        # POSE
        # 33 landmarks × 3 = 99 values
        # ----------------------------------------------------

        if results.pose_landmarks:

            pose = []

            for landmark in results.pose_landmarks.landmark:

                pose.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            pose_detected += 1

        else:

            pose = [0.0] * 99

        # ----------------------------------------------------
        # LEFT HAND
        # 21 landmarks × 3 = 63 values
        # ----------------------------------------------------

        if results.left_hand_landmarks:

            left_hand = []

            for landmark in results.left_hand_landmarks.landmark:

                left_hand.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            left_hand_detected += 1

        else:

            left_hand = [0.0] * 63

        # ----------------------------------------------------
        # RIGHT HAND
        # 21 landmarks × 3 = 63 values
        # ----------------------------------------------------

        if results.right_hand_landmarks:

            right_hand = []

            for landmark in results.right_hand_landmarks.landmark:

                right_hand.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            right_hand_detected += 1

        else:

            right_hand = [0.0] * 63

        # ----------------------------------------------------
        # COMBINE
        #
        # 99 + 63 + 63 = 225
        # ----------------------------------------------------

        frame_features = (
            pose +
            left_hand +
            right_hand
        )

        sequence.append(frame_features)

    # No valid frames
    if len(sequence) == 0:

        return {
            "success": False,
            "frames": 0,
            "pose": 0,
            "left": 0,
            "right": 0
        }

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    data = np.array(
        sequence,
        dtype=np.float32
    )

    # Safety check
    if data.ndim != 2 or data.shape[1] != 225:

        print(
            "ERROR: Unexpected shape:",
            data.shape
        )

        return {
            "success": False,
            "frames": 0,
            "pose": 0,
            "left": 0,
            "right": 0
        }

    # --------------------------------------------------------
    # Create output subfolders automatically
    # --------------------------------------------------------

    output_directory = os.path.dirname(
        output_file
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save .npy
    # --------------------------------------------------------

    np.save(
        output_file,
        data
    )

    return {
        "success": True,
        "frames": data.shape[0],
        "pose": pose_detected,
        "left": left_hand_detected,
        "right": right_hand_detected
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

print()
print("================================================")
print("     BATCH MEDIAPIPE LANDMARK EXTRACTION")
print("================================================")
print()

# ------------------------------------------------------------
# Check dataset
# ------------------------------------------------------------

if not os.path.isdir(DATASET_FOLDER):

    print("ERROR: Dataset folder not found:")
    print(DATASET_FOLDER)
    print()
    print(
        "Change DATASET_FOLDER at the top of this file."
    )

    exit()


# ------------------------------------------------------------
# Find every preprocessed video folder
# ------------------------------------------------------------

print("Searching dataset for frame folders...")
print()

frame_folders = find_frame_folders(
    DATASET_FOLDER
)

print(
    "Total frame folders found:",
    len(frame_folders)
)

if len(frame_folders) == 0:

    print()
    print("ERROR: No JPG/PNG frame folders found.")
    print(
        "Check that DATASET_FOLDER points to your"
        " already-preprocessed dataset."
    )

    exit()


# ------------------------------------------------------------
# Limit first run
# ------------------------------------------------------------

if MAX_VIDEOS is None:

    folders_to_process = frame_folders

else:

    folders_to_process = frame_folders[
        :MAX_VIDEOS
    ]


print(
    "Folders selected for this run:",
    len(folders_to_process)
)

print()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic

processed = 0
skipped = 0
failed = 0

csv_rows = []

start_time = time.time()


with mp_holistic.Holistic(

    static_image_mode=True,

    model_complexity=1,

    min_detection_confidence=0.5

) as holistic:

    # --------------------------------------------------------
    # Process each video folder
    # --------------------------------------------------------

    for number, frames_folder in enumerate(
        folders_to_process,
        start=1
    ):

        print("------------------------------------------------")
        print(
            "Processing:",
            number,
            "/",
            len(folders_to_process)
        )
        print(
            "Frames:",
            frames_folder
        )

        # ----------------------------------------------------
        # Find the path relative to dataset root.
        #
        # Example:
        #
        # Dataset/Greetings/Hello/Video_001
        #
        # becomes:
        #
        # Greetings/Hello/Video_001
        # ----------------------------------------------------

        relative_path = os.path.relpath(
            frames_folder,
            DATASET_FOLDER
        )

        # ----------------------------------------------------
        # Output path:
        #
        # Landmark_Data/
        #     Greetings/
        #         Hello/
        #             Video_001.npy
        # ----------------------------------------------------

        output_file = os.path.join(
            OUTPUT_FOLDER,
            relative_path + ".npy"
        )

        print(
            "Output:",
            output_file
        )

        # ----------------------------------------------------
        # If .npy already exists, don't process it again.
        # ----------------------------------------------------

        if os.path.exists(output_file):

            print(
                "Already exists - SKIPPED"
            )

            # Load it so it can still be included in CSV
            try:

                existing_data = np.load(
                    output_file,
                    mmap_mode="r"
                )

                frame_count = existing_data.shape[0]
                feature_count = existing_data.shape[1]

            except Exception:

                frame_count = ""
                feature_count = ""

            # Get category/class/video from path
            parts = relative_path.replace(
                "\\",
                "/"
            ).split("/")

            category = parts[0] if len(parts) > 0 else ""
            class_name = parts[1] if len(parts) > 1 else ""
            video_name = parts[-1] if len(parts) > 0 else ""

            csv_rows.append([
                category,
                class_name,
                video_name,
                output_file.replace("\\", "/"),
                frame_count,
                feature_count
            ])

            skipped += 1

            continue


        # ----------------------------------------------------
        # Run MediaPipe
        # ----------------------------------------------------

        result = process_video(
            frames_folder,
            output_file,
            holistic
        )

        if not result["success"]:

            print(
                "FAILED"
            )

            failed += 1

            continue


        # ----------------------------------------------------
        # Extract label information from folder structure
        # ----------------------------------------------------

        parts = relative_path.replace(
            "\\",
            "/"
        ).split("/")

        category = (
            parts[0]
            if len(parts) > 0
            else ""
        )

        class_name = (
            parts[1]
            if len(parts) > 1
            else ""
        )

        video_name = (
            parts[-1]
            if len(parts) > 0
            else ""
        )


        # ----------------------------------------------------
        # Add row to CSV
        # ----------------------------------------------------

        csv_rows.append([

            category,

            class_name,

            video_name,

            output_file.replace(
                "\\",
                "/"
            ),

            result["frames"],

            225

        ])


        processed += 1


        # ----------------------------------------------------
        # Print verification information
        # ----------------------------------------------------

        print()
        print(
            "SUCCESS"
        )

        print(
            "Shape:",
            "(",
            result["frames"],
            ", 225 )"
        )

        print(
            "Pose detected:",
            result["pose"],
            "/",
            result["frames"]
        )

        print(
            "Left hand detected:",
            result["left"],
            "/",
            result["frames"]
        )

        print(
            "Right hand detected:",
            result["right"],
            "/",
            result["frames"]
        )

        print()


        # ----------------------------------------------------
        # OPTIONAL frame deletion
        # ----------------------------------------------------

        if DELETE_FRAMES_AFTER_SUCCESS:

            for frame_file in os.listdir(
                frames_folder
            ):

                if frame_file.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):

                    os.remove(
                        os.path.join(
                            frames_folder,
                            frame_file
                        )
                    )

            print(
                "Original JPG frames deleted."
            )


# ============================================================
# SAVE CSV
# ============================================================

csv_file = os.path.join(
    OUTPUT_FOLDER,
    "landmark_labels.csv"
)


os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ------------------------------------------------------------
# Load existing CSV if it already exists
# ------------------------------------------------------------

existing_rows = []

if os.path.exists(csv_file):

    with open(
        csv_file,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.reader(file)

        # Skip header
        next(reader, None)

        for row in reader:
            existing_rows.append(row)


# ------------------------------------------------------------
# Combine old + new rows
# ------------------------------------------------------------

all_rows = existing_rows + csv_rows


# ------------------------------------------------------------
# Remove duplicate entries
# Uses Category + Class + Video as the unique identity
# ------------------------------------------------------------

unique_rows = []
seen = set()

for row in all_rows:

    if len(row) < 3:
        continue

    key = (
        row[0],
        row[1],
        row[2]
    )

    if key not in seen:

        seen.add(key)
        unique_rows.append(row)


# ------------------------------------------------------------
# Save updated CSV
# ------------------------------------------------------------

with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Category",
        "Class",
        "Video",
        "NPY_Path",
        "Frame_Count",
        "Feature_Count"
    ])

    writer.writerows(
        unique_rows
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

elapsed = time.time() - start_time

print()
print("================================================")
print("             EXTRACTION FINISHED")
print("================================================")

print(
    "Processed:",
    processed
)

print(
    "Skipped:",
    skipped
)

print(
    "Failed:",
    failed
)

print(
    "CSV:",
    csv_file
)

print(
    "Time:",
    round(elapsed, 2),
    "seconds"
)

print("================================================")
