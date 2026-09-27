import os
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Masking, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


# =========================
# SETTINGS
# =========================

CSV_FILE = "Landmark_Data/landmark_labels.csv"

NUM_CLASSES = 20
TARGET_FRAMES = 64
FEATURES = 225

BATCH_SIZE = 32
EPOCHS = 50


# =========================
# LANDMARK NORMALIZATION
# =========================

def normalize_landmarks(data):

    data = data.copy()

    # 225 features = 75 landmarks × 3 coordinates
    data = data.reshape(data.shape[0], 75, 3)

    # First 33 landmarks are pose landmarks
    pose = data[:, :33, :]

    # Left and right shoulder
    left_shoulder = pose[:, 11, :]
    right_shoulder = pose[:, 12, :]

    # Center of shoulders
    center = (
        left_shoulder + right_shoulder
    ) / 2.0

    # Move landmarks relative to body center
    data = data - center[:, np.newaxis, :]

    # Shoulder distance for scale normalization
    shoulder_distance = np.linalg.norm(
        left_shoulder - right_shoulder,
        axis=1
    )

    shoulder_distance[
        shoulder_distance < 0.001
    ] = 1.0

    # Scale normalization
    data = data / shoulder_distance[
        :, np.newaxis, np.newaxis
    ]

    return data.reshape(
        data.shape[0],
        225
    )


# =========================
# RESAMPLE SEQUENCE
# =========================

def resample_sequence(data, target_frames):

    old_frames = data.shape[0]

    # If already correct length
    if old_frames == target_frames:
        return data

    # Select evenly spaced frames
    indices = np.linspace(
        0,
        old_frames - 1,
        target_frames
    ).astype(int)

    return data[indices]


# =========================
# LOAD DATASET
# =========================

print("Loading dataset...")

df = pd.read_csv(CSV_FILE)

print("Total samples:", len(df))
print("Total classes:", df["Class"].nunique())


# =========================
# SELECT TOP 20 CLASSES
# =========================

class_counts = df["Class"].value_counts()

selected_classes = class_counts.head(
    NUM_CLASSES
).index.tolist()

print("\nSelected classes:")

for i, class_name in enumerate(
    selected_classes
):

    print(
        i + 1,
        class_name,
        "-",
        class_counts[class_name],
        "samples"
    )


df = df[
    df["Class"].isin(selected_classes)
].copy()

print(
    "\nSamples after selecting classes:",
    len(df)
)


# =========================
# LOAD LANDMARK FILES
# =========================

X = []
y = []

print("\nLoading landmark files...")


for index, row in df.iterrows():

    file_path = row["NPY_Path"].replace(
        "/",
        os.sep
    )

    if not os.path.exists(file_path):

        print(
            "Missing:",
            file_path
        )

        continue

    data = np.load(file_path)

    # Check feature count
    if data.shape[1] != FEATURES:

        print(
            "Wrong feature count:",
            file_path
        )

        continue

    # Normalize landmarks
    data = normalize_landmarks(data)

    # Resample to 64 frames
    data = resample_sequence(
        data,
        TARGET_FRAMES
    )

    X.append(data)
    y.append(row["Class"])


X = np.array(
    X,
    dtype=np.float32
)

y = np.array(y)


print("\nData loaded successfully!")

print(
    "X shape:",
    X.shape
)

print(
    "Number of samples:",
    len(y)
)


# =========================
# ENCODE LABELS
# =========================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print(
    "Number of classes:",
    len(encoder.classes_)
)


# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)


# =========================
# TRAIN / VALIDATION SPLIT
# =========================

X_train, X_val, y_train, y_val = train_test_split(

    X_train,
    y_train,

    test_size=0.10,

    random_state=42,

    stratify=y_train
)


print("\nDataset split:")

print(
    "Training:",
    len(X_train)
)

print(
    "Validation:",
    len(X_val)
)

print(
    "Testing:",
    len(X_test)
)


# =========================
# ONE-HOT ENCODING
# =========================

y_train = to_categorical(
    y_train,
    num_classes=len(
        encoder.classes_
    )
)

y_val = to_categorical(
    y_val,
    num_classes=len(
        encoder.classes_
    )
)

y_test = to_categorical(
    y_test,
    num_classes=len(
        encoder.classes_
    )
)


# =========================
# BUILD LSTM
# =========================

model = Sequential([

    Input(
        shape=(
            TARGET_FRAMES,
            FEATURES
        )
    ),

    Masking(
        mask_value=0.0
    ),

    LSTM(
        128,
        return_sequences=True
    ),

    Dropout(0.3),

    LSTM(64),

    Dropout(0.3),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        len(encoder.classes_),
        activation="softmax"
    )
])


model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


print("\nModel summary:")

model.summary()


# =========================
# CALLBACKS
# =========================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=4,

    min_lr=0.00001
)


# =========================
# TRAIN
# =========================

print(
    "\nStarting normalized 64-frame training...\n"
)


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


# =========================
# TEST
# =========================

print(
    "\nEvaluating model..."
)


test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=1
)


print(
    "\n=============================="
)

print(
    "NORMALIZED 64-FRAME RESULTS"
)

print(
    "=============================="
)


print(
    "Test Loss:",
    test_loss
)

print(
    "Test Accuracy:",
    test_accuracy
)

print(
    "Test Accuracy (%):",
    test_accuracy * 100
)


# =========================
# SAVE MODEL
# =========================

model.save(
    "trained_lstm_20classes_norm64.keras"
)


with open(
    "label_encoder_20classes_norm64.pkl",
    "wb"
) as f:

    pickle.dump(
        encoder,
        f
    )


print(
    "\nModel saved:"
)

print(
    "trained_lstm_20classes_norm64.keras"
)

print(
    "\nLabel encoder saved:"
)

print(
    "label_encoder_20classes_norm64.pkl"
)