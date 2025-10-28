"""
Neural network models for plant classification
"""
from .plant_cnn import PlantCNN, create_ensemble_model

__all__ = ['PlantCNN', 'create_ensemble_model']
