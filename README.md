
# Hand-Recognition-Scratch

Hand-Recognition-Scratch is a Python-based project utilizing MediaPipe, TensorFlow, and OpenCV for real-time hand gesture recognition. The project covers the entire workflow, from data collection and preprocessing to model training and real-time prediction.

## Requirements
- **Language Used**:
   - Python 3.11.9

- **Libraries Used**:
   - Tensorflow 2.18.0
   - Mediapipe 0.10.20
   - Opencv-python 4.10.0.84
   - Keras 3.7.0
   - Scikit-learn 1.6.0

## Features
- **Webcam Testing**: Validate webcam functionality.
- **Module Testing**: Verify MediaPipe module for hand detection.
- **Custom Dataset Generation**: Collect and label gesture images.
- **Data Preprocessing**: Prepares dataset by saving features and labels as `X.npy` and `y.npy` files.
- **Model Training**: Trains a TensorFlow model and saves it as `gesture_recognition_model.keras`.
- **Real-Time Prediction**: Predicts gestures from live webcam feed using the trained model.

## File Overview
- **`webcam.py`**: Tests webcam functionality.
- **`real-time-hand-detection.py`**: Confirms MediaPipe module functionality for hand detection.
- **`data-collection.py`**: Captures gesture images and labels to create a custom dataset.
- **`preprocessing.py`**: Processes the dataset and saves:
  - `X.npy`: Feature data (image pixel values).
  - `y.npy`: Label data (corresponding gesture classes).
- **`train.py`**: Trains the gesture recognition model and saves it as `gesture_recognition_model.keras`.
- **`real-time-prediction.py`**: Uses the trained model to recognize gestures in a live video feed.

## Requirements
Install the required libraries using:

```bash
pip install -r requirements.txt
```

## Usage
1. **Test Webcam**:

   ```bash
   python webcam.py
   ```

2. **Verify MediaPipe Module**:

   ```bash
   python real-time-hand-detection.py
   ```

3. **Collect Dataset**:

   ```bash
   python data-collection.py
   ```

4. **Preprocess Dataset**:

   ```bash
   python preprocessing.py
   ```

5. **Train Model**:

   ```bash
   python train.py
   ```

6. **Real-Time Prediction**:

   ```bash
   python real-time-prediction.py
   ```

## Output of preprocessing.py
- **Preprocessed Dataset**: `X.npy` (features) and `y.npy` (labels).

## Output of train.py
- **Trained Model**: `gesture_recognition_model.keras`.

