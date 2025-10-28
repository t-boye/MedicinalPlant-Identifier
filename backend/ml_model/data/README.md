# Training Data Directory Structure

This directory contains the training data for the medicinal plant identification model.

## Directory Structure

```
data/
├── raw/                    # Original training images
│   ├── plant_class_1/     # One directory per plant species
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── plant_class_2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
├── processed/             # Preprocessed images (auto-generated)
└── augmented/            # Augmented images (auto-generated)
```

## How to Add Training Data

1. **Organize Your Images**: Create one subdirectory per plant species inside the `raw/` directory
   - Directory name will be used as the class name
   - Example: `raw/Aloe_Vera/`, `raw/Turmeric/`, etc.

2. **Image Requirements**:
   - Supported formats: `.jpg`, `.jpeg`, `.png`
   - Recommended: High-quality images (at least 500x500 pixels)
   - Multiple angles and lighting conditions for each plant
   - Minimum: 50-100 images per class (more is better)
   - Maximum: No limit, but balanced classes work best

3. **Best Practices**:
   - Use clear, focused images of the plant
   - Include various parts: leaves, flowers, stems, roots
   - Capture different growth stages
   - Include natural backgrounds
   - Avoid heavily edited or filtered images

## Example Structure

```
raw/
├── Aloe_Vera/
│   ├── aloe_001.jpg
│   ├── aloe_002.jpg
│   ├── aloe_003.jpg
│   └── ... (50+ images)
├── Turmeric/
│   ├── turmeric_001.jpg
│   ├── turmeric_002.jpg
│   └── ... (50+ images)
├── Neem/
│   ├── neem_001.jpg
│   ├── neem_002.jpg
│   └── ... (50+ images)
└── ...
```

## Data Sources

Consider these sources for training data:
- Field photography (best option)
- Plant databases (e.g., iNaturalist, PlantNet)
- Academic datasets
- Government agricultural databases
- Ensure you have proper licenses for all images

## After Adding Data

Once you've added your training data, run:
```bash
cd backend
python ml_model/training/train_model.py --data-dir ml_model/data/raw --balance-classes --epochs 50
```

## Current Status

**Data directory is currently empty. Please add training data before training the model.**

To check your data structure:
```bash
ls -R ml_model/data/raw/
```
