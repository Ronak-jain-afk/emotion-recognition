"""
Real-Time Emotion Recognition using Webcam
Captures video, detects faces, and predicts emotions using trained CNN model
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

IMG_SIZE = (48, 48)


def load_resources(model_path, cascade_path):
    """
    Load the trained emotion model and face detection cascade.
    
    Args:
        model_path: Path to trained Keras model
        cascade_path: Path to Haar Cascade XML file
    
    Returns:
        model, face_cascade
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Please train the model first using: python src/train.py"
        )
    
    print(f"Loading emotion model from: {model_path}")
    model = load_model(model_path)
    
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar Cascade not found at: {cascade_path}")
    
    print(f"Loading face cascade from: {cascade_path}")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    return model, face_cascade


def preprocess_face(face_img):
    """
    Preprocess detected face for emotion prediction.
    
    Steps:
    1. Convert to grayscale (if not already)
    2. Resize to 48x48
    3. Normalize pixel values (0-1)
    4. Reshape for model input
    
    Args:
        face_img: Cropped face image (BGR or grayscale)
    
    Returns:
        Preprocessed face array ready for model prediction
    """
    if len(face_img.shape) == 3:
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_img
    
    face_resized = cv2.resize(face_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    face_normalized = face_resized.astype('float32') / 255.0
    
    face_input = np.expand_dims(face_normalized, axis=0)
    face_input = np.expand_dims(face_input, axis=-1)
    
    return face_input


def draw_emotion_label(frame, x, y, w, h, emotion):
    """
    Draw rectangle around face and emotion label above it.
    
    Args:
        frame: Video frame to draw on
        x, y, w, h: Face bounding box coordinates
        emotion: Predicted emotion label string
    """
    BOX_COLOR = (0, 255, 0)  # Green
    TEXT_COLOR = (255, 255, 255)  # White
    TEXT_BG_COLOR = (0, 255, 0)  # Green background for text
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(
        emotion, font, font_scale, thickness
    )
    
    text_x = x
    text_y = y - 10
    
    if text_y - text_height < 0:
        text_y = y + h + text_height + 10
    
    padding = 5
    cv2.rectangle(
        frame,
        (text_x - padding, text_y - text_height - padding),
        (text_x + text_width + padding, text_y + padding),
        TEXT_BG_COLOR,
        -1  
    )
    
    cv2.putText(
        frame,
        emotion,
        (text_x, text_y),
        font,
        font_scale,
        TEXT_COLOR,
        thickness
    )


def run_emotion_detection(model, face_cascade, camera_index=0):
    """
    Run real-time emotion detection on webcam feed.
    
    Program flow:
    1. Camera captures frame
    2. Face detected using Haar Cascade
    3. Face cropped from frame
    4. Face resized & normalized
    5. CNN predicts emotion
    6. Emotion label shown on screen
    
    Press ESC to exit.
    
    Args:
        model: Trained emotion recognition model
        face_cascade: OpenCV face detection cascade
        camera_index: Camera device index (default: 0)
    """
    print(f"\nOpening camera (index: {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open camera. Please check your webcam connection.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "=" * 50)
    print("EMOTION RECOGNITION STARTED")
    print("=" * 50)
    print("Press ESC to exit")
    print("-" * 50)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            face_input = preprocess_face(face_roi)
            
            predictions = model.predict(face_input, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            emotion = EMOTION_LABELS[emotion_idx]
            
            draw_emotion_label(frame, x, y, w, h, emotion)
        
        cv2.imshow('Emotion Recognition - Press ESC to Exit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("\nESC pressed. Exiting...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")


def main():
    """Main function to run emotion detection."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    model_path = os.path.join(project_dir, 'models', 'emotion_model.keras')
    cascade_path = os.path.join(project_dir, 'models', 'haarcascade_frontalface_default.xml')
    
    print("=" * 50)
    print("REAL-TIME EMOTION RECOGNITION")
    print("=" * 50)
    
    model, face_cascade = load_resources(model_path, cascade_path)
    
    run_emotion_detection(model, face_cascade)


if __name__ == "__main__":
    main()
