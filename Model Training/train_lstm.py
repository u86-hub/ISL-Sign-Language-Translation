import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = r"Landmark_Data\landmark_labels.csv"
BASE_FOLDER = r"Landmark_Data"

MAX_SEQUENCE_LENGTH = 154

EPOCHS = 50
BATCH_SIZE = 32

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# LOAD CSV
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(CSV_PATH)

print("Total samples:", len(df))
print("Total classes:", df["Class"].nunique())

# Remove possible invalid rows
df = df.dropna(subset=["NPY_Path", "Class"])

# ============================================================
# LOAD NPY FILES
# ============================================================

sequences = []
labels = []

print("\nLoading landmark sequences...")

for i, row in df.iterrows():

    file_path = row["NPY_Path"].replace("/", os.sep)

    if not os.path.exists(file_path):
        print("Missing file:", file_path)
        continue

    try:
        data = np.load(file_path)

        # Make sure shape is correct
        if data.ndim != 2 or data.shape[1] != 225:
            print("Skipping wrong shape:", file_path, data.shape)
            continue

        sequences.append(data)
        labels.append(row["Class"])

    except Exception as e:
        print("Could not load:", file_path)
        print(e)

    if (i + 1) % 500 == 0:
        print("Loaded:", i + 1, "/", len(df))


print("\nSuccessfully loaded:", len(sequences))


# ============================================================
# LABEL ENCODING
# ============================================================

print("\nEncoding labels...")

label_encoder = LabelEncoder()

labels_encoded = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)

print("Number of classes:", num_classes)

print("\nFirst 10 classes:")
for i in range(min(10, num_classes)):
    print(i, "=", label_encoder.classes_[i])


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nCreating train/validation/test split...")

# Create indices for each class
class_indices = {}

for index, label in enumerate(labels_encoded):

    if label not in class_indices:
        class_indices[label] = []

    class_indices[label].append(index)


train_indices = []
val_indices = []
test_indices = []

rng = np.random.default_rng(RANDOM_SEED)

for label, indices in class_indices.items():

    indices = np.array(indices)

    rng.shuffle(indices)

    n = len(indices)

    if n >= 5:

        # 70% train, 15% validation, 15% test
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])

    elif n == 4:

        # 2 train, 1 validation, 1 test
        train_indices.extend(indices[:2])
        val_indices.append(indices[2])
        test_indices.append(indices[3])

    elif n == 3:

        # 1 train, 1 validation, 1 test
        train_indices.append(indices[0])
        val_indices.append(indices[1])
        test_indices.append(indices[2])

    elif n == 2:

        # 1 train, 1 test
        train_indices.append(indices[0])
        test_indices.append(indices[1])

    else:

        # Only one sample
        # Keep it in training
        train_indices.append(indices[0])


print("Training samples:", len(train_indices))
print("Validation samples:", len(val_indices))
print("Testing samples:", len(test_indices))


# ============================================================
# CREATE DATASETS
# ============================================================

X_train = [sequences[i] for i in train_indices]
X_val = [sequences[i] for i in val_indices]
X_test = [sequences[i] for i in test_indices]

y_train = labels_encoded[train_indices]
y_val = labels_encoded[val_indices]
y_test = labels_encoded[test_indices]


# ============================================================
# PAD SEQUENCES
# ============================================================

print("\nPadding sequences...")

X_train = pad_sequences(
    X_train,
    maxlen=MAX_SEQUENCE_LENGTH,
    dtype="float32",
    padding="post",
    truncating="post",
    value=0.0
)

X_val = pad_sequences(
    X_val,
    maxlen=MAX_SEQUENCE_LENGTH,
    dtype="float32",
    padding="post",
    truncating="post",
    value=0.0
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_SEQUENCE_LENGTH,
    dtype="float32",
    padding="post",
    truncating="post",
    value=0.0
)


print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)


# ============================================================
# ONE-HOT ENCODE LABELS
# ============================================================

y_train = to_categorical(
    y_train,
    num_classes=num_classes
)

y_val = to_categorical(
    y_val,
    num_classes=num_classes
)

y_test = to_categorical(
    y_test,
    num_classes=num_classes
)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

print("\nBuilding LSTM model...")

model = Sequential([

    Masking(
        mask_value=0.0,
        input_shape=(MAX_SEQUENCE_LENGTH, 225)
    ),

    LSTM(
        128,
        return_sequences=True
    ),

    Dropout(0.3),

    LSTM(
        64
    ),

    Dropout(0.3),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        num_classes,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nMODEL SUMMARY")
print("=" * 60)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

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


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 60)
print("STARTING LSTM TRAINING")
print("=" * 60)

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
    ],

    verbose=1
)


# ============================================================
# TEST MODEL
# ============================================================

print("\n")
print("=" * 60)
print("TESTING MODEL")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# SAVE MODEL
# ============================================================

model.save("trained_lstm.keras")

joblib.dump(
    label_encoder,
    "label_encoder.pkl"
)

print("\nModel saved as:")
print("trained_lstm.keras")

print("\nLabel encoder saved as:")
print("label_encoder.pkl")


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("LSTM Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.savefig(
    "accuracy_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig(
    "loss_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

actual_labels = np.argmax(
    y_test,
    axis=1
)


cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=np.arange(num_classes)
)

plt.figure(
    figsize=(30, 30)
)

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    cbar=True
)

plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    rotation=90,
    fontsize=5
)

plt.yticks(
    fontsize=5
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=np.arange(num_classes),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


print("\n")
print("=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

print("\nFiles generated:")

print("1. trained_lstm.keras")
print("2. label_encoder.pkl")
print("3. accuracy_graph.png")
print("4. loss_graph.png")
print("5. confusion_matrix.png")