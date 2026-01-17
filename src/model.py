"""
CNN Model Architecture for Emotion Recognition
Lightweight 4-block CNN optimized for real-time inference on 48x48 grayscale images
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, Dropout,
    Flatten, Dense, Input
)


def create_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Create a lightweight CNN for emotion recognition.
    
    Architecture:
    - 4 Convolutional blocks (32 -> 64 -> 128 -> 256 filters)
    - Each block: Conv2D -> BatchNorm -> Conv2D -> BatchNorm -> MaxPool -> Dropout
    - Dense layers: Flatten -> Dense(512) -> Dropout -> Dense(7, softmax)
    
    Args:
        input_shape: Input image shape (default: 48x48x1 grayscale)
        num_classes: Number of emotion classes (default: 7)
    
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        Input(shape=input_shape),
        
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def get_model_summary():
    """Print model summary for debugging."""
    model = create_emotion_model()
    model.summary()
    return model


if __name__ == "__main__":
    model = get_model_summary()
    print(f"\nTotal parameters: {model.count_params():,}")
