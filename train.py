import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Load preprocessed data
X = np.load("X.npy")
y = np.load("y.npy")

# Validate labels
unique_labels = np.unique(y)
print(f"Unique labels in y: {unique_labels}")
assert np.min(unique_labels) >= 0, "Negative labels found!"
assert np.max(unique_labels) < 4, f"Unexpected label value: {np.max(unique_labels)}"

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights to handle class imbalance
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Add regularization to prevent overfitting
    layers.Dense(4, activation='softmax')  # Update to 4 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use categorical loss for multi-class classification
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=15,
    validation_data=(X_val, y_val),
    class_weight=class_weights,  # Handle class imbalance
    batch_size=32
)

# Save the trained model
model.save("gesture_recognition_model.keras")
print("Model trained and saved successfully.")
