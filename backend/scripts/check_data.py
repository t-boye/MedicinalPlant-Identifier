"""
Script to check training data structure and statistics
"""
import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.logging_config import get_logger

logger = get_logger(__name__)


def check_training_data(data_dir: str = "./ml_model/data/raw"):
    """
    Check the training data structure and provide statistics

    Args:
        data_dir: Path to the raw data directory
    """
    data_path = Path(data_dir)

    logger.info("=" * 70)
    logger.info("Training Data Check")
    logger.info("=" * 70)

    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return False

    # Get all subdirectories (classes)
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    if not class_dirs:
        logger.warning("\n⚠ No training data found!")
        logger.info(f"\nPlease add your training data to: {data_path.absolute()}")
        logger.info("\nExpected structure:")
        logger.info("  raw/")
        logger.info("    ├── plant_class_1/")
        logger.info("    │   ├── image1.jpg")
        logger.info("    │   └── image2.jpg")
        logger.info("    ├── plant_class_2/")
        logger.info("    │   └── ...")
        logger.info("\nSee ml_model/data/README.md for more details.")
        return False

    # Analyze each class
    logger.info(f"\nFound {len(class_dirs)} plant classes:")
    logger.info("-" * 70)

    class_stats = []
    total_images = 0
    supported_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name

        # Count images
        images = [f for f in class_dir.iterdir()
                  if f.is_file() and f.suffix in supported_extensions]
        num_images = len(images)
        total_images += num_images

        class_stats.append((class_name, num_images))

        # Status indicator
        if num_images < 30:
            status = "⚠ TOO FEW"
        elif num_images < 50:
            status = "⚡ LOW"
        elif num_images < 100:
            status = "✓ GOOD"
        else:
            status = "✓✓ EXCELLENT"

        logger.info(f"  {status:12} | {class_name:30} | {num_images:5} images")

    logger.info("-" * 70)
    logger.info(f"\nTotal classes: {len(class_dirs)}")
    logger.info(f"Total images: {total_images}")
    logger.info(f"Average images per class: {total_images / len(class_dirs):.1f}")

    # Check class balance
    image_counts = [count for _, count in class_stats]
    min_count = min(image_counts)
    max_count = max(image_counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    logger.info(f"\nClass balance:")
    logger.info(f"  Min images: {min_count}")
    logger.info(f"  Max images: {max_count}")
    logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}x")

    if imbalance_ratio > 3:
        logger.warning("\n⚠ Classes are significantly imbalanced!")
        logger.info("  Consider using --balance-classes flag during training")

    # Recommendations
    logger.info("\n" + "=" * 70)
    logger.info("Recommendations:")
    logger.info("=" * 70)

    if min_count < 50:
        logger.warning("⚠ Some classes have very few images (<50)")
        logger.info("  → Try to collect more images for better model performance")

    if imbalance_ratio > 3:
        logger.warning("⚠ Significant class imbalance detected")
        logger.info("  → Use --balance-classes flag to augment minority classes")

    if total_images < 500:
        logger.warning("⚠ Relatively small dataset")
        logger.info("  → Consider using transfer learning (default) for better results")

    logger.info("\nTo train the model, run:")
    logger.info(f"  python ml_model/training/train_model.py --data-dir {data_dir} --balance-classes --epochs 50")

    logger.info("\n" + "=" * 70)

    return True


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Check training data structure')
    parser.add_argument('--data-dir', type=str, default='./ml_model/data/raw',
                        help='Path to raw data directory')

    args = parser.parse_args()

    success = check_training_data(args.data_dir)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
