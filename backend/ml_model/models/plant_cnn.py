"""
CNN Model for Medicinal Plant Identification
Uses transfer learning with EfficientNetB0 and custom classification head
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, MobileNetV2
from tensorflow.keras.regularizers import l2


class PlantCNN:
    """
    CNN model for plant identification with support for transfer learning
    and bias mitigation through balanced training
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple = (224, 224, 3),
        base_model_name: str = 'efficientnet',
        dropout_rate: float = 0.3,
        l2_regularization: float = 0.01
    ):
        """
        Initialize the plant CNN model

        Args:
            num_classes: Number of plant species to classify
            input_shape: Input image shape (height, width, channels)
            base_model_name: Base model for transfer learning
            dropout_rate: Dropout rate for regularization
            l2_regularization: L2 regularization factor
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.base_model_name = base_model_name
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_regularization
        self.model = None

    def build_model(self, trainable_base: bool = False):
        """
        Build the CNN model with transfer learning

        Args:
            trainable_base: Whether to fine-tune the base model

        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = keras.Input(shape=self.input_shape)

        # Data augmentation layers (applied during training)
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)
        x = layers.RandomContrast(0.2)(x)

        # Load pre-trained base model
        base_model = self._get_base_model()
        base_model.trainable = trainable_base

        # If fine-tuning, only train the last few layers
        if trainable_base:
            base_model.trainable = True
            # Freeze all layers except the last 20
            for layer in base_model.layers[:-20]:
                layer.trainable = False

        # Base model
        x = base_model(x, training=False)

        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)

        # Dense layers with regularization
        x = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=l2(self.l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=l2(self.l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Output layer with softmax for multi-class classification
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            kernel_regularizer=l2(self.l2_reg),
            name='predictions'
        )(x)

        # Create model
        self.model = keras.Model(inputs, outputs, name='plant_cnn')

        return self.model

    def _get_base_model(self):
        """Load the appropriate base model"""
        base_models = {
            'efficientnet': EfficientNetB0,
            'resnet': ResNet50V2,
            'mobilenet': MobileNetV2,
        }

        if self.base_model_name.lower() not in base_models:
            raise ValueError(f"Unknown base model: {self.base_model_name}")

        BaseModel = base_models[self.base_model_name.lower()]

        return BaseModel(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )

    def compile_model(
        self,
        learning_rate: float = 0.001,
        use_class_weights: bool = True
    ):
        """
        Compile the model with optimizer and loss function

        Args:
            learning_rate: Learning rate for optimizer
            use_class_weights: Whether to use class weights for balanced training
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Use Adam optimizer with learning rate schedule
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Compile with categorical crossentropy and metrics
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
            ]
        )

    def get_callbacks(self, model_save_path: str):
        """
        Get training callbacks for better model performance

        Args:
            model_save_path: Path to save the best model

        Returns:
            List of Keras callbacks
        """
        callbacks = [
            # Save best model
            keras.callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),

            # Early stopping to prevent overfitting
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),

            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_graph=True
            )
        ]

        return callbacks

    def summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        self.model.summary()


def create_ensemble_model(models_list, num_classes):
    """
    Create an ensemble model from multiple trained models
    for better accuracy and reduced bias

    Args:
        models_list: List of trained models
        num_classes: Number of classes

    Returns:
        Ensemble model
    """
    inputs = keras.Input(shape=(224, 224, 3))

    # Get predictions from all models
    outputs = [model(inputs) for model in models_list]

    # Average predictions
    avg_output = layers.Average()(outputs)

    # Create ensemble model
    ensemble = keras.Model(inputs=inputs, outputs=avg_output, name='ensemble_model')

    return ensemble
