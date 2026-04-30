import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# ================= SETTINGS =================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
DATASET_PATH = "dataset"

MODEL_PATH = "model.h5"
CLASSES_PATH = "classes.json"

# ================= DATA AUGMENTATION =================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ================= SAVE CLASS NAMES =================
class_names = list(train_data.class_indices.keys())

with open(CLASSES_PATH, "w") as f:
    json.dump(class_names, f)

print("✅ Classes:", class_names)

# ================= MODEL =================
model = models.Sequential([

    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation='softmax')
])

# ================= COMPILE =================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================= TRAIN =================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ================= SAVE MODEL SAFELY =================
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)

model.save(MODEL_PATH)

print("✅ Model saved successfully:", MODEL_PATH)