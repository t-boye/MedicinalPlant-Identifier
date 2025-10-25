"""
Data preprocessing and augmentation utilities for plant dataset
Includes bias mitigation through balanced sampling and diverse augmentation
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras


class PlantDataPreprocessor:
    """
    Preprocessor for plant image data with bias mitigation
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        augmentation_strength: str = 'medium'
    ):
        """
        Initialize the data preprocessor

        Args:
            image_size: Target image size (height, width)
            augmentation_strength: 'light', 'medium', or 'heavy'
        """
        self.image_size = image_size
        self.augmentation_strength = augmentation_strength
        self.augmentation_pipeline = self._create_augmentation_pipeline()

    def _create_augmentation_pipeline(self):
        """
        Create data augmentation pipeline using Albumentations
        Diverse augmentations help reduce bias
        """
        if self.augmentation_strength == 'light':
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ]
        elif self.augmentation_strength == 'medium':
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=30, p=0.6),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ]
        else:  # heavy
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=45, p=0.7),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.3,
                    rotate_limit=45,
                    p=0.6
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.4,
                    contrast_limit=0.4,
                    p=0.6
                ),
                A.HueSaturationValue(
                    hue_shift_limit=30,
                    sat_shift_limit=40,
                    val_shift_limit=30,
                    p=0.6
                ),
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(p=1.0),
                    A.MotionBlur(p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.GridDistortion(p=1.0),
                    A.ElasticTransform(p=1.0),
                ], p=0.3),
                A.CLAHE(p=0.3),
                A.RandomShadow(p=0.2),
            ]

        return A.Compose(transforms)

    def load_and_preprocess_image(
        self,
        image_path: str,
        augment: bool = False
    ) -> np.ndarray:
        """
        Load and preprocess a single image

        Args:
            image_path: Path to the image file
            augment: Whether to apply augmentation

        Returns:
            Preprocessed image array
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, self.image_size)

        # Apply augmentation if requested
        if augment:
            augmented = self.augmentation_pipeline(image=image)
            image = augmented['image']

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        return image

    def load_dataset_from_directory(
        self,
        data_dir: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load dataset from directory structure:
        data_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                img2.jpg

        Args:
            data_dir: Root directory containing class subdirectories
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (images, labels, metadata)
        """
        data_dir = Path(data_dir)
        images = []
        labels = []
        class_names = []

        # Get all class directories
        class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)}

        print(f"Found {len(class_dirs)} classes")

        # Load images from each class
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_names.append(class_name)
            class_idx = class_to_idx[class_name]

            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.jpeg')) + \
                         list(class_dir.glob('*.png'))

            print(f"Loading {len(image_files)} images from class '{class_name}'")

            for img_file in image_files:
                try:
                    img = self.load_and_preprocess_image(str(img_file), augment=False)
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")

        images = np.array(images)
        labels = np.array(labels)

        # Split into train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
        )

        metadata = {
            'num_classes': len(class_names),
            'class_names': class_names,
            'class_to_idx': class_to_idx,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
        }

        return (X_train, y_train), (X_val, y_val), (X_test, y_test), metadata

    @staticmethod
    def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for balanced training
        Critical for bias mitigation

        Args:
            labels: Array of class labels

        Returns:
            Dictionary mapping class index to weight
        """
        classes = np.unique(labels)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels
        )

        class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

        print("Class weights computed:")
        for cls, weight in class_weights.items():
            print(f"  Class {cls}: {weight:.4f}")

        return class_weights

    @staticmethod
    def create_tf_dataset(
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = False
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset with optional augmentation

        Args:
            images: Image array
            labels: Label array
            batch_size: Batch size
            shuffle: Whether to shuffle
            augment: Whether to apply augmentation

        Returns:
            TensorFlow Dataset
        """
        # Convert labels to categorical
        num_classes = len(np.unique(labels))
        labels_categorical = keras.utils.to_categorical(labels, num_classes)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels_categorical))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


def balance_dataset_by_augmentation(
    images: np.ndarray,
    labels: np.ndarray,
    target_samples_per_class: int,
    preprocessor: PlantDataPreprocessor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance dataset by augmenting minority classes
    Helps reduce bias towards majority classes

    Args:
        images: Image array
        labels: Label array
        target_samples_per_class: Target number of samples per class
        preprocessor: PlantDataPreprocessor instance

    Returns:
        Balanced images and labels
    """
    balanced_images = []
    balanced_labels = []

    classes = np.unique(labels)

    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        cls_images = images[cls_indices]
        cls_count = len(cls_images)

        # Add original images
        balanced_images.extend(cls_images)
        balanced_labels.extend([cls] * cls_count)

        # Augment if needed
        if cls_count < target_samples_per_class:
            num_augment = target_samples_per_class - cls_count
            aug_indices = np.random.choice(cls_indices, size=num_augment, replace=True)

            for idx in aug_indices:
                # Apply augmentation
                augmented = preprocessor.augmentation_pipeline(image=images[idx])
                balanced_images.append(augmented['image'])
                balanced_labels.append(cls)

            print(f"Class {cls}: {cls_count} -> {target_samples_per_class} (augmented {num_augment})")
        else:
            print(f"Class {cls}: {cls_count} samples (no augmentation needed)")

    balanced_images = np.array(balanced_images)
    balanced_labels = np.array(balanced_labels)

    # Shuffle
    shuffle_idx = np.random.permutation(len(balanced_images))
    balanced_images = balanced_images[shuffle_idx]
    balanced_labels = balanced_labels[shuffle_idx]

    return balanced_images, balanced_labels
