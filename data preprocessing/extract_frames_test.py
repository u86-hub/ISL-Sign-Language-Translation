import cv2
import os

# Path to the test video
video_path = r"C:\Users\nivedita\OneDrive\Desktop\ISL_Project\Selected_Dataset\Greetings\Hello\MVI_0029.MOV"

# Output folder
output_folder = r"Preprocessed_Frames\Hello\MVI_0029"

os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # -----------------------------------------
    # Resize while keeping aspect ratio
    # -----------------------------------------

    height, width = frame.shape[:2]

    scale = min(224 / width, 224 / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    frame = cv2.resize(
        frame,
        (new_width, new_height)
    )

    # -----------------------------------------
    # Add padding to make 224 x 224
    # -----------------------------------------

    top = (224 - new_height) // 2
    bottom = 224 - new_height - top

    left = (224 - new_width) // 2
    right = 224 - new_width - left

    frame = cv2.copyMakeBorder(
        frame,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    # -----------------------------------------
    # Save frame
    # -----------------------------------------

    filename = f"frame_{frame_number:04d}.jpg"

    output_path = os.path.join(
        output_folder,
        filename
    )

    cv2.imwrite(output_path, frame)

    frame_number += 1

cap.release()

print("Frame extraction completed!")
print("Total frames saved:", frame_number)