import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size and augmentation settings
image_size = (64, 64)
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)

# Define gestures and initialize dataset
gesture_labels = {"thumbs_up": 0, "palm_open": 1, "swipe_right": 2, "no_gesture": 3}
dataset = []
labels = []

for gesture, label in gesture_labels.items():
    folder = f"dataset/{gesture}"
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        image = cv2.imread(filepath)

        # Ensure valid image
        if image is None:
            print(f"Failed to read {filepath}, skipping.")
            continue

        # Resize and normalize
        resized_image = cv2.resize(image, image_size)
        normalized_image = resized_image / 255.0
        dataset.append(normalized_image)
        labels.append(label)

        # Augment data
        augmented = datagen.flow(
            np.expand_dims(resized_image, axis=0),
            batch_size=1
        )
        for _ in range(3):  # Generate 3 augmented samples per image
            augmented_image = next(augmented)[0] / 255.0
            dataset.append(augmented_image)
            labels.append(label)

# Convert to numpy arrays
X = np.array(dataset)
y = np.array(labels)

# Save the preprocessed data
np.save("X.npy", X)
np.save("y.npy", y)
print(f"Dataset saved. X shape: {X.shape}, y shape: {y.shape}")
