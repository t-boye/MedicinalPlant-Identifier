"""
Script to initialize the database
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import init_db, check_db_connection
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def main():
    """Initialize the database"""
    logger.info("=" * 60)
    logger.info("Database Initialization Script")
    logger.info("=" * 60)

    # Check database connection
    logger.info("\nChecking database connection...")
    if not check_db_connection():
        logger.error("Database connection failed. Please check your DATABASE_URL in .env")
        logger.error("Make sure PostgreSQL is running and the database exists.")
        return 1

    # Initialize database
    logger.info("\nCreating database tables...")
    try:
        init_db()
        logger.info("\n" + "=" * 60)
        logger.info("Database initialization complete!")
        logger.info("=" * 60)
        return 0
    except Exception as e:
        logger.error(f"\nDatabase initialization failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
