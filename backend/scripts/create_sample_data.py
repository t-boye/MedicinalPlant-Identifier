"""
Create sample training data for testing
This creates synthetic images to test the training pipeline
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.logging_config import get_logger

logger = get_logger(__name__)


def create_sample_plant_image(plant_name: str, image_id: int, size=(224, 224)):
    """
    Create a synthetic plant image for testing

    Args:
        plant_name: Name of the plant
        image_id: Image identifier
        size: Image size (width, height)

    Returns:
        PIL Image
    """
    # Create a colorful image
    img = Image.new('RGB', size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Use different colors for different plants
    colors = {
        'Aloe_Vera': [(34, 139, 34), (50, 205, 50), (0, 128, 0)],
        'Turmeric': [(255, 215, 0), (255, 165, 0), (218, 165, 32)],
        'Neem': [(85, 107, 47), (107, 142, 35), (124, 252, 0)],
        'Basil': [(0, 100, 0), (34, 139, 34), (0, 128, 0)],
        'Mint': [(152, 251, 152), (144, 238, 144), (0, 250, 154)]
    }

    plant_colors = colors.get(plant_name, [(100, 150, 100), (120, 170, 120), (80, 130, 80)])

    # Draw random shapes to simulate leaves/plant parts
    for _ in range(random.randint(10, 20)):
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        r = random.randint(20, 60)
        color = random.choice(plant_colors)

        # Random shape variation
        shape_type = random.choice(['ellipse', 'polygon'])
        if shape_type == 'ellipse':
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=color)
        else:
            points = [(x + random.randint(-r, r), y + random.randint(-r, r)) for _ in range(6)]
            draw.polygon(points, fill=color, outline=color)

    # Add some texture
    for _ in range(100):
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        draw.point((x, y), fill=random.choice(plant_colors))

    # Add plant name text (small)
    try:
        # Try to use a font, fall back to default if not available
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    text = f"{plant_name}_{image_id}"
    draw.text((5, 5), text, fill=(255, 255, 255), font=font)

    # Add some random variations
    img = img.rotate(random.randint(-15, 15))

    return img


def create_sample_dataset(
    output_dir: str = "./ml_model/data/raw",
    num_classes: int = 5,
    images_per_class: int = 30
):
    """
    Create a sample dataset for testing

    Args:
        output_dir: Output directory for the dataset
        num_classes: Number of plant classes
        images_per_class: Number of images per class
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default plant classes
    plant_classes = ['Aloe_Vera', 'Turmeric', 'Neem', 'Basil', 'Mint']
    plant_classes = plant_classes[:num_classes]

    logger.info("=" * 70)
    logger.info("Creating Sample Training Dataset")
    logger.info("=" * 70)
    logger.info(f"\nOutput directory: {output_path.absolute()}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Images per class: {images_per_class}")
    logger.info(f"Total images: {num_classes * images_per_class}")

    total_created = 0

    for plant_name in plant_classes:
        class_dir = output_path / plant_name
        class_dir.mkdir(exist_ok=True)

        logger.info(f"\nCreating images for {plant_name}...")

        for i in range(images_per_class):
            # Create synthetic image
            img = create_sample_plant_image(plant_name, i)

            # Save image
            img_path = class_dir / f"{plant_name.lower()}_{i:03d}.jpg"
            img.save(img_path, 'JPEG', quality=85)
            total_created += 1

            if (i + 1) % 10 == 0:
                logger.info(f"  Created {i + 1}/{images_per_class} images")

    logger.info("\n" + "=" * 70)
    logger.info("Sample Dataset Created Successfully!")
    logger.info("=" * 70)
    logger.info(f"\nTotal images created: {total_created}")
    logger.info(f"Dataset location: {output_path.absolute()}")
    logger.info("\n⚠️  NOTE: These are SYNTHETIC images for testing only!")
    logger.info("For a real model, use actual plant photographs.")
    logger.info("\nNext steps:")
    logger.info("1. Check data: python scripts/check_data.py")
    logger.info("2. Train model: python scripts/train.py --quick-test --balance-classes")
    logger.info("=" * 70)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Create sample training data for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default sample dataset (5 classes, 30 images each)
  python scripts/create_sample_data.py

  # Create small test dataset (3 classes, 20 images each)
  python scripts/create_sample_data.py --num-classes 3 --images-per-class 20

  # Create larger sample dataset
  python scripts/create_sample_data.py --num-classes 5 --images-per-class 50

NOTE: This creates SYNTHETIC images for testing the pipeline only.
      For a real model, use actual plant photographs.
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./ml_model/data/raw',
        help='Output directory for sample data'
    )

    parser.add_argument(
        '--num-classes',
        type=int,
        default=5,
        help='Number of plant classes (default: 5)'
    )

    parser.add_argument(
        '--images-per-class',
        type=int,
        default=30,
        help='Number of images per class (default: 30)'
    )

    args = parser.parse_args()

    # Check if directory already has data
    output_path = Path(args.output_dir)
    if output_path.exists():
        existing_dirs = [d for d in output_path.iterdir() if d.is_dir()]
        if existing_dirs:
            logger.warning(f"⚠️  Directory {output_path} already contains {len(existing_dirs)} subdirectories")
            response = input("Do you want to continue and add more data? (y/n): ")
            if response.lower() != 'y':
                logger.info("Cancelled.")
                return 0

    try:
        create_sample_dataset(
            output_dir=args.output_dir,
            num_classes=args.num_classes,
            images_per_class=args.images_per_class
        )
        return 0
    except Exception as e:
        logger.error(f"Failed to create sample dataset: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
