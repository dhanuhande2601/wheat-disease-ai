import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128

# =========================
# Load Dataset/.
# =========================
def load_data(data_dir):
    data = []
    labels = []
    
    categories = sorted(os.listdir(data_dir))
    print("Classes:", categories)

    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)

            if img_array is None:
                continue

            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(img_array)
            labels.append(label)

    return np.array(data), np.array(labels), categories

# =========================
# Load Data
# =========================
data_dir = "dataset"

print("Loading dataset...")
X, y, classes = load_data(data_dir)

X = X / 255.0

print("Total images:", len(X))

# =========================
# Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Data Augmentation 🔥
# =========================
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

# =========================
# Build Strong Model
# =========================
model = models.Sequential()

model.add(tf.keras.Input(shape=(128, 128, 3)))

model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Dropout(0.5))  # overfitting reduce

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))

# 🔥 dynamic classes
model.add(layers.Dense(len(classes), activation='softmax'))

# =========================
# Compile
# =========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# Callbacks (SMART TRAINING)
# =========================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =========================
# Train Model
# =========================
print("Training started...")

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=5000,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# =========================
# Save Model + Classes
# =========================
model.save("wheat_disease_model.keras")

# Save class names (VERY IMPORTANT)
np.save("classes.npy", classes)

print("✅ Training completed and model saved!")