"""
Simplified training script wrapper
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml_model.training.train_model import PlantModelTrainer
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def main():
    """Main training function with simplified interface"""
    parser = argparse.ArgumentParser(
        description='Train Medicinal Plant Identification Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training with default settings
  python scripts/train.py

  # Custom training with specific parameters
  python scripts/train.py --epochs 100 --batch-size 64 --base-model resnet

  # Training with class balancing and fine-tuning
  python scripts/train.py --balance-classes --fine-tune
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./ml_model/data/raw',
        help='Directory containing training data (default: ./ml_model/data/raw)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./saved_models',
        help='Directory to save trained models (default: ./saved_models)'
    )

    parser.add_argument(
        '--base-model',
        type=str,
        default='efficientnet',
        choices=['efficientnet', 'resnet', 'mobilenet'],
        help='Base model architecture (default: efficientnet)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Initial learning rate (default: 0.001)'
    )

    parser.add_argument(
        '--balance-classes',
        action='store_true',
        help='Balance classes through augmentation (recommended for imbalanced datasets)'
    )

    parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='Perform fine-tuning after initial training (improves accuracy but takes longer)'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test run with 5 epochs (for testing)'
    )

    args = parser.parse_args()

    # Override epochs for quick test
    if args.quick_test:
        args.epochs = 5
        logger.info("Quick test mode: Training for only 5 epochs")

    # Check if data exists
    data_path = Path(args.data_dir)
    if not data_path.exists() or not list(data_path.iterdir()):
        logger.error("=" * 70)
        logger.error("ERROR: No training data found!")
        logger.error("=" * 70)
        logger.error(f"\nPlease add training data to: {data_path.absolute()}")
        logger.error("\nSee ml_model/data/README.md for instructions on how to structure your data.")
        logger.error("\nYou can also run: python scripts/check_data.py")
        return 1

    # Display training configuration
    logger.info("=" * 70)
    logger.info("Training Configuration")
    logger.info("=" * 70)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Balance classes: {args.balance_classes}")
    logger.info(f"Fine-tune: {args.fine_tune}")
    logger.info("=" * 70)

    try:
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
        logger.info("\n" + "=" * 70)
        logger.info("Evaluating model on test set...")
        logger.info("=" * 70)
        trainer.evaluate()

        # Fine-tune if requested
        if args.fine_tune:
            logger.info("\n" + "=" * 70)
            logger.info("Starting fine-tuning...")
            logger.info("=" * 70)
            fine_tuned_path = trainer.fine_tune(epochs=20, learning_rate=0.0001)
            trainer.evaluate()

        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("Training Complete!")
        logger.info("=" * 70)
        logger.info(f"\nModel saved to: {model_path}")
        logger.info("\nTo use this model:")
        logger.info(f"1. Update MODEL_PATH in .env file to: {model_path}")
        logger.info("2. Start the API server: uvicorn app.main:app --reload")
        logger.info("3. Test the API at: http://localhost:8000/docs")
        logger.info("\n" + "=" * 70)

        return 0

    except Exception as e:
        logger.error(f"\n\nTraining failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
