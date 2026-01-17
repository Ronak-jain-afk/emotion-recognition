"""
Data Preprocessing and Loading Utilities for Emotion Recognition
Uses ImageDataGenerator for loading FER2013 dataset with augmentation
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

IMG_SIZE = (48, 48)
INPUT_SHAPE = (48, 48, 1)
NUM_CLASSES = 7


def get_data_generators(train_dir, test_dir, batch_size=64):
    """
    Create data generators for training and validation.
    
    Training augmentation includes:
    - Rescaling (1/255)
    - Rotation (±15°)
    - Width/Height shift (10%)
    - Horizontal flip
    - Zoom (10%)
    
    Validation only applies rescaling.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test/validation data directory
        batch_size: Batch size for training (default: 64)
    
    Returns:
        train_generator, validation_generator
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator


def get_class_weights(train_generator):
    """
    Calculate class weights to handle imbalanced dataset.
    FER2013 has imbalanced classes (e.g., 'Disgust' has fewer samples).
    
    Args:
        train_generator: Training data generator
    
    Returns:
        Dictionary of class weights
    """
    import numpy as np
    from collections import Counter
    
    class_counts = Counter(train_generator.classes)
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (num_classes * count)
    
    return class_weights


def verify_data_structure(data_dir):
    """
    Verify the data directory structure is correct.
    
    Expected structure:
    data/
    ├── train/
    │   ├── Angry/
    │   ├── Disgust/
    │   ├── Fear/
    │   ├── Happy/
    │   ├── Sad/
    │   ├── Surprise/
    │   └── Neutral/
    └── test/
        └── (same subfolders)
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        True if structure is valid, False otherwise
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found: {train_dir}")
        return False
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        return False
    
    train_classes = set(os.listdir(train_dir))
    test_classes = set(os.listdir(test_dir))
    
    print(f"Training classes found: {sorted(train_classes)}")
    print(f"Test classes found: {sorted(test_classes)}")
    
    print("\nTraining data distribution:")
    total_train = 0
    for emotion in sorted(train_classes):
        emotion_dir = os.path.join(train_dir, emotion)
        if os.path.isdir(emotion_dir):
            count = len([f for f in os.listdir(emotion_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            total_train += count
            print(f"  {emotion}: {count} images")
    
    print(f"\nTotal training images: {total_train}")
    
    print("\nTest data distribution:")
    total_test = 0
    for emotion in sorted(test_classes):
        emotion_dir = os.path.join(test_dir, emotion)
        if os.path.isdir(emotion_dir):
            count = len([f for f in os.listdir(emotion_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            total_test += count
            print(f"  {emotion}: {count} images")
    
    print(f"\nTotal test images: {total_test}")
    
    return True


if __name__ == "__main__":
    import sys
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    
    print(f"Verifying data structure in: {data_dir}\n")
    verify_data_structure(data_dir)
