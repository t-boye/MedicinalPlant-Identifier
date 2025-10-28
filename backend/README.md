# Medicinal Plant Identifier - Backend

FastAPI backend for identifying medicinal plants using deep learning with EfficientNetB0 transfer learning.

## 🌿 Current Status

✅ **Fully Operational**
- Trained model with 6 medicinal plant classes (63% test accuracy)
- API server running on http://localhost:8000
- CORS configured for frontend integration
- Comprehensive logging and error handling

## Features

- **Plant Identification**: CNN-based classification using EfficientNetB0 transfer learning
- **RESTful API**: FastAPI with automatic Swagger documentation
- **Trained Model**: 6 medicinal plant classes with class balancing
- **Bias Mitigation**: Class balancing and data augmentation using Albumentations
- **Production Ready**: Proper logging, error handling, and environment configuration

## 🎯 Trained Plant Classes

The current model can identify these 6 medicinal plants:
1. **Arjun Leaf** (230 images)
2. **Curry Leaf** (230 images)
3. **Marsh Pennywort Leaf** (230 images)
4. **Mint Leaf** (230 images)
5. **Neem Leaf** (230 images)
6. **Rubble Leaf** (230 images)

**Dataset**: 1,380 images total (well-balanced across all classes)
**Location**: `app/dataset/Original Images (Version 02)/`

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Edit `.env` and update the following:
- `DATABASE_URL`: Your PostgreSQL connection string
- `SECRET_KEY`: Generate a secure key (use `openssl rand -hex 32`)
- Other settings as needed

### 3. Initialize Database (Optional)

If you want to use the database features:

```bash
python scripts/init_db.py
```

### 4. Training Data

**Current Dataset** (already included):
```
app/dataset/Original Images (Version 02)/
├── Arjun_Leaf/     (230 images)
├── Curry_Leaf/     (230 images)
├── Marsh_Pennywort_Leaf/  (230 images)
├── Mint_Leaf/      (230 images)
├── Neem_Leaf/      (230 images)
└── Rubble_Leaf/    (230 images)
```

**For custom training data**, organize in `ml_model/data/raw/`:
```
ml_model/data/raw/
├── Plant_Species_1/
│   ├── image1.jpg
│   └── ...
└── Plant_Species_2/
    └── ...
```

See `ml_model/data/README.md` for detailed instructions.

### 5. Check Your Data

```bash
python scripts/check_data.py
```

This will show statistics about your training data and recommendations.

### 6. Train the Model

**Current Model** (already trained):
- Model: `plant_cnn_efficientnet_20251028_025622.keras`
- Architecture: EfficientNetB0 with custom classification head
- Test Accuracy: 63.04%
- Classes: 6 medicinal plants
- Training: 5 epochs (quick test)

**Quick test training**:
```bash
python scripts/train.py --data-dir "app/dataset/Original Images (Version 02)" --quick-test --balance-classes
```

**Full training for better accuracy**:
```bash
python scripts/train.py \
  --data-dir "app/dataset/Original Images (Version 02)" \
  --base-model efficientnet \
  --epochs 30 \
  --batch-size 32 \
  --balance-classes
```

**Best quality training**:
```bash
python scripts/train.py \
  --data-dir "app/dataset/Original Images (Version 02)" \
  --base-model efficientnet \
  --epochs 50 \
  --batch-size 32 \
  --balance-classes \
  --fine-tune
```

Training options:
- `--data-dir`: Path to training data directory
- `--base-model`: Choose from `efficientnet`, `resnet`, or `mobilenet`
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32, reduce to 16 if out of memory)
- `--learning-rate`: Initial learning rate (default: 0.001)
- `--balance-classes`: Balance classes through augmentation (recommended)
- `--fine-tune`: Perform fine-tuning after initial training
- `--quick-test`: Quick 5-epoch test run for validation

### 7. Update Model Path

**Current configuration** (already set in `.env`):
```env
MODEL_PATH=./saved_models/plant_cnn_efficientnet_20251028_025622.keras
```

After retraining, update the `MODEL_PATH` in your `.env` file:
```env
MODEL_PATH=./saved_models/your_new_model_name.keras
```

### 8. Run the API Server

**Development** (with auto-reload):
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Production**:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The server will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 API Documentation

Once the server is running, access interactive documentation:
- **Swagger UI**: http://localhost:8000/docs (Try out API endpoints)
- **ReDoc**: http://localhost:8000/redoc (Alternative documentation view)

## 🔌 API Endpoints

### Plant Identification (Working)

**Identify Plant from Image**
```http
POST /api/recognition/identify
Content-Type: multipart/form-data

Response:
{
  "predictions": [
    {
      "class_name": "Mint_Leaf",
      "confidence": 0.85,
      "class_id": 3
    }
  ],
  "top_prediction": {...},
  "model_info": {
    "model_type": "efficientnet",
    "num_classes": 6,
    "confidence_threshold": 0.3
  }
}
```

**Get Model Information**
```http
GET /api/recognition/model/info
```

**Get Plant Classes**
```http
GET /api/recognition/model/classes

Response:
{
  "num_classes": 6,
  "classes": ["Arjun_Leaf", "Curry_Leaf", ...]
}
```

### Plant Database (TODO - Future Enhancement)

- `GET /api/plants/`: Get all plants
- `GET /api/plants/{id}`: Get specific plant details
- `GET /api/plants/search`: Search plants

### Example Request (Python)

```python
import requests

url = "http://localhost:8000/api/recognition/identify"
files = {'file': open('plant_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

### Example Request (JavaScript)

```javascript
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/api/recognition/identify', {
  method: 'POST',
  body: formData
});
const result = await response.json();
console.log(result.predictions);
```

## Project Structure

```
backend/
├── app/
│   ├── api/              # API endpoints
│   │   ├── auth.py
│   │   ├── plants.py
│   │   └── recognition.py
│   ├── core/             # Core configuration
│   │   ├── config.py
│   │   ├── database.py
│   │   └── logging_config.py
│   ├── models/           # Database models
│   │   └── plant.py
│   └── main.py          # FastAPI application
├── ml_model/
│   ├── models/          # Neural network architectures
│   │   └── plant_cnn.py
│   ├── training/        # Training scripts
│   │   └── train_model.py
│   ├── utils/           # ML utilities
│   │   ├── data_preprocessing.py
│   │   ├── inference.py
│   │   └── model_evaluation.py
│   └── data/            # Training data
│       ├── raw/         # Original images
│       ├── processed/
│       └── augmented/
├── scripts/             # Helper scripts
│   ├── check_data.py   # Check training data
│   ├── init_db.py      # Initialize database
│   └── train.py        # Training wrapper
├── alembic/            # Database migrations
├── saved_models/       # Trained models
├── logs/               # Application logs
└── requirements.txt
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
```

### Linting

```bash
flake8 .
```

## ⚙️ Configuration

### Environment Variables (.env)

```env
# API Settings
API_V1_STR=/api/v1
PROJECT_NAME=Medicinal Plant Identifier

# CORS - Frontend URLs (JSON array format)
ALLOWED_ORIGINS=["http://localhost:5178","http://localhost:5173","http://localhost:3000"]

# ML Model
MODEL_PATH=./saved_models/plant_cnn_efficientnet_20251028_025622.keras
MODEL_INPUT_SIZE=[224,224]
CONFIDENCE_THRESHOLD=0.3

# Server
HOST=0.0.0.0
PORT=8000
RELOAD=True

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

**Important**:
- `ALLOWED_ORIGINS` must be in JSON array format
- `MODEL_INPUT_SIZE` must be in JSON array format
- Update `MODEL_PATH` after training a new model

## 🐛 Troubleshooting

### CORS errors from frontend
- Check that frontend URL is in `ALLOWED_ORIGINS` in `.env`
- Format must be JSON array: `["http://localhost:5178"]`
- Restart the server after changing `.env`

### Model not found error
- Verify model path in `.env` exists
- Check `saved_models/` directory for trained models
- Train a model first: `python scripts/train.py --quick-test`

### 422 Unprocessable Entity on /identify
- Frontend must send image with field name `'file'` (not `'image'`)
- Check Content-Type header: `multipart/form-data`
- Verify image file is valid (PNG, JPG, JPEG)

### Out of memory during training
- Reduce `--batch-size` to 16 or 8
- Use smaller base model: `--base-model mobilenet`
- Close other memory-intensive applications
- Use GPU if available

### Import errors
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)
- Create virtual environment if needed

### TensorFlow warnings (oneDNN)
- These are informational, not errors
- Can be ignored safely
- To disable: `export TF_ENABLE_ONEDNN_OPTS=0`

## License

MIT License
