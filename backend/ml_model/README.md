# Medicinal Plant Identification ML Model

A CNN-based deep learning model for identifying medicinal plants with bias mitigation techniques.

## Features

- **Transfer Learning**: Uses pre-trained models (EfficientNetB0, ResNet50V2, MobileNetV2) as base
- **Bias Mitigation**: Multiple techniques to ensure fair and balanced predictions
  - Class balancing through augmentation
  - Weighted loss function
  - Per-class performance monitoring
- **Robust Augmentation**: Diverse augmentation pipeline to improve generalization
- **Ensemble Support**: Multiple models can be combined for better accuracy
- **Comprehensive Evaluation**: Detailed metrics and bias detection

## Architecture

### Base Models Supported
- **EfficientNetB0** (Default) - Best balance of accuracy and speed
- **ResNet50V2** - High accuracy, larger model
- **MobileNetV2** - Fast inference, mobile-friendly

### Custom Layers
- Global Average Pooling
- Dense layers (512, 256) with BatchNormalization
- Dropout (0.3) for regularization
- L2 regularization
- Softmax output layer

## Training

### Prepare Your Data

Organize your dataset in this structure:
```
data/
├── AloeVera/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Neem/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── Turmeric/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

### Train the Model

```bash
cd backend/ml_model/training

# Basic training
python train_model.py --data-dir /path/to/data --output-dir ../saved_models

# With bias mitigation (recommended)
python train_model.py \
    --data-dir /path/to/data \
    --output-dir ../saved_models \
    --balance-classes \
    --epochs 50 \
    --batch-size 32

# With fine-tuning for better accuracy
python train_model.py \
    --data-dir /path/to/data \
    --output-dir ../saved_models \
    --balance-classes \
    --fine-tune \
    --base-model efficientnet
```

### Training Parameters

- `--data-dir`: Directory containing training images
- `--output-dir`: Where to save trained models (default: ./saved_models)
- `--base-model`: Base architecture (efficientnet/resnet/mobilenet)
- `--batch-size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--learning-rate`: Initial learning rate (default: 0.001)
- `--balance-classes`: Enable class balancing through augmentation
- `--fine-tune`: Perform fine-tuning after initial training

## Bias Mitigation Techniques

### 1. Class Balancing
- Augments minority classes to match majority class size
- Prevents model from being biased toward common plants

### 2. Class Weights
- Computes balanced class weights during training
- Penalizes misclassification of minority classes more heavily

### 3. Diverse Augmentation
- Multiple augmentation techniques (rotation, flip, brightness, etc.)
- Reduces overfitting to specific image characteristics

### 4. Bias Detection
- Monitors per-class performance during evaluation
- Flags classes with significantly lower accuracy
- Provides recommendations for improvement

## Evaluation

The model provides comprehensive evaluation metrics:

- **Overall Metrics**: Accuracy, Precision, Recall, F1-Score
- **Per-Class Metrics**: Individual performance for each plant
- **Confusion Matrix**: Visual representation of predictions
- **Bias Analysis**: Detects if model is biased toward certain classes

## Inference

### Using the PlantIdentifier

```python
from ml_model.utils.inference import PlantIdentifier

# Load model
identifier = PlantIdentifier(
    model_path='saved_models/plant_cnn_efficientnet.keras',
    confidence_threshold=0.5
)

# Predict from file
predictions = identifier.predict_from_file('plant_image.jpg', top_k=5)

for pred in predictions:
    print(f"{pred['class_name']}: {pred['confidence_percentage']:.2f}%")
```

### Using Ensemble

```python
from ml_model.utils.inference import EnsembleIdentifier

# Load multiple models
ensemble = EnsembleIdentifier(
    model_paths=[
        'saved_models/model1.keras',
        'saved_models/model2.keras',
        'saved_models/model3.keras'
    ]
)

# Predict with voting
predictions = ensemble.predict(image, voting='soft')
```

## API Integration

The model is integrated with the FastAPI backend:

```bash
# Start the backend server
cd backend
uvicorn app.main:app --reload
```

### API Endpoints

- `POST /api/recognition/identify` - Identify plant from image
- `GET /api/recognition/model/info` - Get model information
- `GET /api/recognition/model/classes` - Get list of plant classes

## Model Files

After training, you'll have:
- `plant_cnn_*.keras` - Trained model weights
- `plant_cnn_*_metadata.json` - Model metadata and class names
- `logs/` - TensorBoard training logs

## Requirements

- TensorFlow 2.15.0
- scikit-learn 1.3.2
- OpenCV 4.8.1
- Albumentations 1.3.1
- NumPy 1.24.3
- Matplotlib 3.8.2

## Tips for Better Results

1. **Diverse Dataset**: Include images from different angles, lighting, backgrounds
2. **Balanced Classes**: Aim for similar number of images per plant species
3. **High Quality Images**: Use clear, well-lit images
4. **Data Augmentation**: Enable --balance-classes flag during training
5. **Fine-Tuning**: Use --fine-tune for additional accuracy boost
6. **Ensemble**: Combine multiple models for more robust predictions

## Bias Mitigation Best Practices

1. Always use `--balance-classes` flag during training
2. Review bias analysis report after evaluation
3. If bias detected:
   - Collect more data for underperforming classes
   - Increase augmentation strength
   - Try different base models
   - Use ensemble of models

## Troubleshooting

### Model accuracy is low
- Check if dataset is balanced
- Try different base model (efficientnet vs resnet)
- Increase training epochs
- Enable fine-tuning

### Model is biased toward certain classes
- Use --balance-classes flag
- Collect more data for minority classes
- Review class weights in training logs

### Out of memory during training
- Reduce batch size
- Use MobileNetV2 as base model
- Reduce image size in preprocessing

## Future Improvements

- [ ] Multi-scale input processing
- [ ] Attention mechanisms
- [ ] Self-supervised pre-training
- [ ] Active learning for data collection
- [ ] Real-time inference optimization
- [ ] Mobile deployment (TFLite conversion)
