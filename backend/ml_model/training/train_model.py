"""
Training script for medicinal plant identification model
Includes bias mitigation techniques and comprehensive evaluation
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.plant_cnn import PlantCNN
from utils.data_preprocessing import (
    PlantDataPreprocessor,
    balance_dataset_by_augmentation
)


class PlantModelTrainer:
    """
    Trainer for plant identification model with bias mitigation
    """

    def __init__(
        self,
        data_dir: str,
        model_save_dir: str = './saved_models',
        image_size: Tuple[int, int] = (224, 224),
        base_model: str = 'efficientnet',
        batch_size: int = 32,
        epochs: int = 50,
        learning_rate: float = 0.001
    ):
        """
        Initialize the trainer

        Args:
            data_dir: Directory containing training data
            model_save_dir: Directory to save trained models
            image_size: Input image size
            base_model: Base model architecture
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Initial learning rate
        """
        self.data_dir = Path(data_dir)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        self.image_size = image_size
        self.base_model = base_model
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.preprocessor = PlantDataPreprocessor(
            image_size=image_size,
            augmentation_strength='medium'
        )

        self.model = None
        self.history = None
        self.metadata = None
        self.class_weights = None

    def load_and_prepare_data(
        self,
        balance_classes: bool = True,
        target_samples: int = None
    ):
        """
        Load and prepare training data with bias mitigation

        Args:
            balance_classes: Whether to balance classes through augmentation
            target_samples: Target number of samples per class (if balancing)
        """
        print("=" * 60)
        print("Loading and preparing dataset...")
        print("=" * 60)

        # Load dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test), self.metadata = \
            self.preprocessor.load_dataset_from_directory(
                str(self.data_dir),
                test_size=0.2,
                val_size=0.1
            )

        print(f"\nDataset loaded:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Number of classes: {self.metadata['num_classes']}")

        # Check class distribution
        print(f"\nClass distribution in training set:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            class_name = self.metadata['class_names'][cls]
            print(f"  {class_name} (Class {cls}): {count} samples")

        # Balance classes if requested (BIAS MITIGATION)
        if balance_classes:
            print("\n" + "=" * 60)
            print("Balancing dataset through augmentation (BIAS MITIGATION)")
            print("=" * 60)

            if target_samples is None:
                target_samples = int(np.max(counts) * 1.2)  # 120% of max class

            X_train, y_train = balance_dataset_by_augmentation(
                X_train,
                y_train,
                target_samples_per_class=target_samples,
                preprocessor=self.preprocessor
            )

            print(f"\nBalanced dataset: {len(X_train)} training samples")

        # Compute class weights (ADDITIONAL BIAS MITIGATION)
        print("\n" + "=" * 60)
        print("Computing class weights for balanced training")
        print("=" * 60)
        self.class_weights = self.preprocessor.compute_class_weights(y_train)

        # Create TensorFlow datasets
        self.train_dataset = self.preprocessor.create_tf_dataset(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True,
            augment=True
        )

        self.val_dataset = self.preprocessor.create_tf_dataset(
            X_val, y_val,
            batch_size=self.batch_size,
            shuffle=False,
            augment=False
        )

        self.test_dataset = self.preprocessor.create_tf_dataset(
            X_test, y_test,
            batch_size=self.batch_size,
            shuffle=False,
            augment=False
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def build_and_compile_model(self):
        """Build and compile the model"""
        print("\n" + "=" * 60)
        print("Building model...")
        print("=" * 60)

        cnn = PlantCNN(
            num_classes=self.metadata['num_classes'],
            input_shape=(*self.image_size, 3),
            base_model_name=self.base_model,
            dropout_rate=0.3,
            l2_regularization=0.01
        )

        self.model = cnn.build_model(trainable_base=False)
        cnn.compile_model(learning_rate=self.learning_rate)

        print("\nModel architecture:")
        cnn.summary()

        self.cnn = cnn

    def train(self):
        """Train the model"""
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)

        # Generate unique model name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"plant_cnn_{self.base_model}_{timestamp}"
        model_path = self.model_save_dir / f"{model_name}.keras"

        # Get callbacks
        callbacks = self.cnn.get_callbacks(str(model_path))

        # Train model with class weights (BIAS MITIGATION)
        self.history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=self.class_weights,  # Critical for bias mitigation
            verbose=1
        )

        print(f"\nModel saved to: {model_path}")

        # Save metadata
        metadata_path = self.model_save_dir / f"{model_name}_metadata.json"
        self._save_metadata(metadata_path)

        return model_path

    def fine_tune(self, epochs: int = 20, learning_rate: float = 0.0001):
        """
        Fine-tune the model by unfreezing base model layers

        Args:
            epochs: Number of fine-tuning epochs
            learning_rate: Fine-tuning learning rate
        """
        print("\n" + "=" * 60)
        print("Fine-tuning model...")
        print("=" * 60)

        # Rebuild model with trainable base
        cnn = PlantCNN(
            num_classes=self.metadata['num_classes'],
            input_shape=(*self.image_size, 3),
            base_model_name=self.base_model,
            dropout_rate=0.3,
            l2_regularization=0.01
        )

        # Build with trainable base (last 20 layers)
        self.model = cnn.build_model(trainable_base=True)
        cnn.compile_model(learning_rate=learning_rate)

        # Generate model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"plant_cnn_{self.base_model}_finetuned_{timestamp}"
        model_path = self.model_save_dir / f"{model_name}.keras"

        # Get callbacks
        callbacks = cnn.get_callbacks(str(model_path))

        # Continue training
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )

        print(f"\nFine-tuned model saved to: {model_path}")

        return model_path

    def evaluate(self, dataset=None):
        """
        Evaluate the model

        Args:
            dataset: Dataset to evaluate on (default: test set)
        """
        if dataset is None:
            dataset = self.test_dataset

        print("\n" + "=" * 60)
        print("Evaluating model...")
        print("=" * 60)

        results = self.model.evaluate(dataset, verbose=1)

        print("\nEvaluation results:")
        for metric_name, value in zip(self.model.metrics_names, results):
            print(f"  {metric_name}: {value:.4f}")

        return results

    def _save_metadata(self, filepath: Path):
        """Save training metadata"""
        metadata = {
            'num_classes': self.metadata['num_classes'],
            'class_names': self.metadata['class_names'],
            'class_to_idx': self.metadata['class_to_idx'],
            'image_size': self.image_size,
            'base_model': self.base_model,
            'training_params': {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
            },
            'class_weights': self.class_weights,
            'training_date': datetime.now().isoformat(),
        }

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to: {filepath}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train plant identification model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='./saved_models',
                       help='Directory to save trained models')
    parser.add_argument('--base-model', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet', 'mobilenet'],
                       help='Base model architecture')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--balance-classes', action='store_true',
                       help='Balance classes through augmentation')
    parser.add_argument('--fine-tune', action='store_true',
                       help='Perform fine-tuning after initial training')

    args = parser.parse_args()

    # Create trainer
    trainer = PlantModelTrainer(
        data_dir=args.data_dir,
        model_save_dir=args.output_dir,
        base_model=args.base_model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )

    # Load and prepare data
    trainer.load_and_prepare_data(balance_classes=args.balance_classes)

    # Build and compile model
    trainer.build_and_compile_model()

    # Train
    model_path = trainer.train()

    # Evaluate
    trainer.evaluate()

    # Fine-tune if requested
    if args.fine_tune:
        fine_tuned_path = trainer.fine_tune(epochs=20, learning_rate=0.0001)
        trainer.evaluate()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
