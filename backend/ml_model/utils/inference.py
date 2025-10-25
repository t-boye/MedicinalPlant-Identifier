"""
Inference utilities for plant identification
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io


class PlantIdentifier:
    """
    Plant identification inference engine
    """

    def __init__(
        self,
        model_path: str,
        metadata_path: str = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the plant identifier

        Args:
            model_path: Path to saved model
            metadata_path: Path to metadata JSON (auto-detected if None)
            confidence_threshold: Minimum confidence for predictions
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold

        # Load metadata
        if metadata_path is None:
            # Auto-detect metadata path
            metadata_path = self.model_path.parent / f"{self.model_path.stem}_metadata.json"

        self.metadata = self._load_metadata(metadata_path)

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(str(model_path))
        print("Model loaded successfully!")

        self.image_size = tuple(self.metadata['image_size'])
        self.class_names = self.metadata['class_names']
        self.num_classes = self.metadata['num_classes']

    def _load_metadata(self, metadata_path: Path) -> Dict:
        """Load metadata from JSON file"""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata

    def preprocess_image(
        self,
        image: np.ndarray,
        resize: bool = True
    ) -> np.ndarray:
        """
        Preprocess image for inference

        Args:
            image: Input image (BGR or RGB)
            resize: Whether to resize image

        Returns:
            Preprocessed image ready for model
        """
        # Convert BGR to RGB if needed (OpenCV uses BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR if values are in 0-255 range
            if image.max() > 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        if resize:
            image = cv2.resize(image, self.image_size)

        # Normalize to [0, 1]
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0

        return image

    def predict(
        self,
        image: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Predict plant species from image

        Args:
            image: Input image
            top_k: Number of top predictions to return

        Returns:
            List of predictions with class names, confidence scores, and indices
        """
        # Preprocess
        processed_image = self.preprocess_image(image)

        # Add batch dimension
        batch_image = np.expand_dims(processed_image, axis=0)

        # Predict
        predictions = self.model.predict(batch_image, verbose=0)[0]

        # Get top-k predictions
        top_indices = np.argsort(predictions)[::-1][:top_k]

        results = []
        for idx in top_indices:
            confidence = float(predictions[idx])

            # Only include predictions above threshold
            if confidence >= self.confidence_threshold:
                results.append({
                    'class_name': self.class_names[idx],
                    'class_id': int(idx),
                    'confidence': confidence,
                    'confidence_percentage': confidence * 100
                })

        return results

    def predict_from_file(
        self,
        image_path: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Predict from image file path

        Args:
            image_path: Path to image file
            top_k: Number of top predictions

        Returns:
            List of predictions
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.predict(image, top_k=top_k)

    def predict_from_bytes(
        self,
        image_bytes: bytes,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Predict from image bytes (useful for API endpoints)

        Args:
            image_bytes: Image data as bytes
            top_k: Number of top predictions

        Returns:
            List of predictions
        """
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)

        # Convert RGBA to RGB if needed
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return self.predict(image, top_k=top_k)

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_path': str(self.model_path),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'image_size': self.image_size,
            'confidence_threshold': self.confidence_threshold,
            'training_date': self.metadata.get('training_date', 'Unknown'),
            'base_model': self.metadata.get('base_model', 'Unknown'),
        }


class EnsembleIdentifier:
    """
    Ensemble of multiple models for more robust predictions
    """

    def __init__(
        self,
        model_paths: List[str],
        metadata_path: str = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize ensemble identifier

        Args:
            model_paths: List of paths to saved models
            metadata_path: Path to metadata (uses first model's metadata)
            confidence_threshold: Minimum confidence threshold
        """
        self.identifiers = []

        for model_path in model_paths:
            identifier = PlantIdentifier(
                model_path=model_path,
                metadata_path=metadata_path,
                confidence_threshold=confidence_threshold
            )
            self.identifiers.append(identifier)

        # Use first model's metadata
        self.metadata = self.identifiers[0].metadata
        self.class_names = self.identifiers[0].class_names
        self.num_classes = self.identifiers[0].num_classes

    def predict(
        self,
        image: np.ndarray,
        top_k: int = 5,
        voting: str = 'soft'
    ) -> List[Dict]:
        """
        Predict using ensemble of models

        Args:
            image: Input image
            top_k: Number of top predictions
            voting: 'soft' (average probabilities) or 'hard' (majority vote)

        Returns:
            Ensemble predictions
        """
        all_predictions = []

        # Get predictions from all models
        for identifier in self.identifiers:
            preds = identifier.predict(image, top_k=self.num_classes)
            all_predictions.append(preds)

        if voting == 'soft':
            # Average probabilities across models
            avg_confidences = np.zeros(self.num_classes)

            for preds in all_predictions:
                for pred in preds:
                    avg_confidences[pred['class_id']] += pred['confidence']

            avg_confidences /= len(self.identifiers)

            # Get top-k
            top_indices = np.argsort(avg_confidences)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append({
                    'class_name': self.class_names[idx],
                    'class_id': int(idx),
                    'confidence': float(avg_confidences[idx]),
                    'confidence_percentage': float(avg_confidences[idx] * 100)
                })

            return results

        else:  # hard voting
            # Count votes for each class
            votes = np.zeros(self.num_classes)

            for preds in all_predictions:
                if preds:  # If there are predictions
                    # Vote for top prediction
                    votes[preds[0]['class_id']] += 1

            # Get top-k
            top_indices = np.argsort(votes)[::-1][:top_k]

            results = []
            for idx in top_indices:
                vote_confidence = votes[idx] / len(self.identifiers)
                results.append({
                    'class_name': self.class_names[idx],
                    'class_id': int(idx),
                    'confidence': float(vote_confidence),
                    'confidence_percentage': float(vote_confidence * 100),
                    'votes': int(votes[idx])
                })

            return results
