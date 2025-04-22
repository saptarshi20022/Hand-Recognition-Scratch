import tensorflow as tf
from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

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

# # Training and validation metrics
# train_acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']

# # Calculate final metrics
# final_train_acc = train_acc[-1]  # Last epoch training accuracy
# final_val_acc = val_acc[-1]      # Last epoch validation accuracy
# final_train_loss = train_loss[-1]
# final_val_loss = val_loss[-1]

# # Print metrics
# print(f"Final Training Accuracy: {final_train_acc:.2f}")
# print(f"Final Validation Accuracy: {final_val_acc:.2f}")
# print(f"Final Training Loss: {final_train_loss:.2f}")
# print(f"Final Validation Loss: {final_val_loss:.2f}")

# # Plot metrics for visualization
# epochs = range(1, len(train_acc) + 1)

# # Accuracy plot
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(epochs, train_acc, label='Training Accuracy')
# plt.plot(epochs, val_acc, label='Validation Accuracy')
# plt.title('Accuracy Over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# # Loss plot
# plt.subplot(1, 2, 2)
# plt.plot(epochs, train_loss, label='Training Loss')
# plt.plot(epochs, val_loss, label='Validation Loss')
# plt.title('Loss Over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()




# # Predictions on the validation set
# y_val_pred = np.argmax(model.predict(X_val), axis=1)

# # Generate classification report
# gesture_labels = {0: "thumbs_up", 1: "palm_open", 2: "swipe_right", 3: "no_gesture"}  # Define labels
# print(classification_report(y_val, y_val_pred, target_names=list(gesture_labels.values())))
