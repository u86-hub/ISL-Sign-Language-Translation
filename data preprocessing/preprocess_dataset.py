import cv2
import os
import csv

# ============================================================
# PATHS
# ============================================================

INPUT_FOLDER = r"Selected_Dataset"
OUTPUT_FOLDER = r"Preprocessed_Frames"

# Final frame size
IMAGE_SIZE = 224


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# FUNCTION TO PREPROCESS ONE VIDEO
# ============================================================

def process_video(video_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("Could not open:", video_path)
        return 0


    frame_number = 0


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # ----------------------------------------------------
        # Keep original aspect ratio
        # ----------------------------------------------------

        height, width = frame.shape[:2]

        scale = min(
            IMAGE_SIZE / width,
            IMAGE_SIZE / height
        )

        new_width = int(width * scale)
        new_height = int(height * scale)


        # Resize
        frame = cv2.resize(
            frame,
            (new_width, new_height)
        )


        # ----------------------------------------------------
        # Add padding to make 224 x 224
        # ----------------------------------------------------

        top = (IMAGE_SIZE - new_height) // 2
        bottom = IMAGE_SIZE - new_height - top

        left = (IMAGE_SIZE - new_width) // 2
        right = IMAGE_SIZE - new_width - left


        frame = cv2.copyMakeBorder(
            frame,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )


        # ----------------------------------------------------
        # Save frame
        # ----------------------------------------------------

        filename = f"frame_{frame_number:04d}.jpg"

        output_path = os.path.join(
            output_folder,
            filename
        )

        cv2.imwrite(output_path, frame)

        frame_number += 1


    cap.release()

    return frame_number


# ============================================================
# PROCESS ALL CLASSES AND VIDEOS
# ============================================================

labels = []

video_extensions = (
    ".mov",
    ".mp4",
    ".avi",
    ".mkv",
    ".webm"
)


for category in os.listdir(INPUT_FOLDER):

    category_path = os.path.join(
        INPUT_FOLDER,
        category
    )

    if not os.path.isdir(category_path):
        continue


    print("\n====================================")
    print("CATEGORY:", category)
    print("====================================")


    # --------------------------------------------------------
    # Go through each class
    # --------------------------------------------------------

    for class_name in os.listdir(category_path):

        class_path = os.path.join(
            category_path,
            class_name
        )

        if not os.path.isdir(class_path):
            continue


        print("\nClass:", class_name)


        # Output folder for this class
        class_output = os.path.join(
            OUTPUT_FOLDER,
            category,
            class_name
        )

        os.makedirs(
            class_output,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Process every video
        # ----------------------------------------------------

        for video_file in os.listdir(class_path):

            if not video_file.lower().endswith(
                video_extensions
            ):
                continue


            video_path = os.path.join(
                class_path,
                video_file
            )


            # Remove file extension
            video_name = os.path.splitext(
                video_file
            )[0]


            # Output folder for this video
            video_output = os.path.join(
                class_output,
                video_name
            )


            print(
                "Processing:",
                video_file
            )


            # Process video
            frame_count = process_video(
                video_path,
                video_output
            )


            # ------------------------------------------------
            # Save label information
            # ------------------------------------------------

            labels.append([
                category,
                class_name,
                video_name,
                frame_count
            ])


# ============================================================
# SAVE LABELS
# ============================================================

labels_file = "labels.csv"

with open(
    labels_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Category",
        "Class",
        "Video",
        "Frame_Count"
    ])

    writer.writerows(labels)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("====================================")
print("PREPROCESSING COMPLETED!")
print("====================================")

print(
    "Total videos processed:",
    len(labels)
)

print(
    "Labels saved to:",
    labels_file
)