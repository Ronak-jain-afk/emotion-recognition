"""
Training Script for Emotion Recognition CNN
Trains the model on FER2013 dataset with early stopping and model checkpointing
"""

import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model import create_emotion_model
from preprocess import get_data_generators, get_class_weights, verify_data_structure


def train_model(
    data_dir,
    model_save_path,
    epochs=60,
    batch_size=64,
    learning_rate=0.001,
    patience=10
):
    """
    Train the emotion recognition model.
    
    Args:
        data_dir: Path to data directory containing train/ and test/ folders
        model_save_path: Path to save the trained model
        epochs: Maximum number of training epochs (default: 60)
        batch_size: Training batch size (default: 64)
        learning_rate: Initial learning rate for Adam optimizer (default: 0.001)
        patience: Early stopping patience (default: 10)
    
    Returns:
        Training history object
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if not verify_data_structure(data_dir):
        raise ValueError("Invalid data directory structure. Please check the data folder.")
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    print("\nLoading data...")
    train_generator, validation_generator = get_data_generators(
        train_dir, test_dir, batch_size=batch_size
    )
    
    class_weights = get_class_weights(train_generator)
    print(f"\nClass weights: {class_weights}")
    
    print("\nCreating model...")
    model = create_emotion_model()
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print(f"\nTraining for up to {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Early stopping patience: {patience}")
    print("-" * 60)
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    
    print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"\nModel saved to: {model_save_path}")
    
    return history


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Training history object from model.fit()
        save_path: Optional path to save the plot image
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main training function."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_dir, 'data')
    models_dir = os.path.join(project_dir, 'models')
    
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, 'emotion_model.keras')
    plot_save_path = os.path.join(models_dir, 'training_history.png')
    
    EPOCHS = 60
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    PATIENCE = 10
    
    history = train_model(
        data_dir=data_dir,
        model_save_path=model_save_path,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE
    )
    
    plot_training_history(history, save_path=plot_save_path)


if __name__ == "__main__":
    main()
