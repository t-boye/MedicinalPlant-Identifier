"""
Recognition API endpoints for plant identification
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
from pathlib import Path
import sys

# Add ml_model to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'ml_model'))

from ml_model.utils.inference import PlantIdentifier
from app.models.plant import MedicinalPlant
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

router = APIRouter()

# Global model instance (loaded on startup)
plant_identifier: Optional[PlantIdentifier] = None


def get_plant_identifier():
    """Get the global plant identifier instance"""
    global plant_identifier

    if plant_identifier is None:
        # Load model
        model_path = os.getenv('MODEL_PATH', settings.MODEL_PATH)

        logger.info(f"Loading model from: {model_path}")

        if not os.path.exists(model_path):
            error_msg = f"Model not found at {model_path}. Please train the model first."
            logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )

        try:
            plant_identifier = PlantIdentifier(
                model_path=model_path,
                confidence_threshold=settings.CONFIDENCE_THRESHOLD
            )
            logger.info(f"Model loaded successfully with {plant_identifier.num_classes} classes")
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )

    return plant_identifier


@router.post("/identify")
async def identify_plant(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Identify a medicinal plant from an uploaded image

    Args:
        file: Uploaded image file
        top_k: Number of top predictions to return

    Returns:
        Identification results with confidence scores
    """
    logger.info(f"Received plant identification request for file: {file.filename}")

    # Validate file type
    if not file.content_type.startswith('image/'):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    try:
        # Read image bytes
        image_bytes = await file.read()
        logger.debug(f"Image size: {len(image_bytes)} bytes")

        # Get identifier
        identifier = get_plant_identifier()

        # Predict
        predictions = identifier.predict_from_bytes(image_bytes, top_k=top_k)

        if not predictions:
            logger.info("No confident predictions found")
            return JSONResponse(
                status_code=200,
                content={
                    "message": "No confident predictions. Image may not contain a recognizable medicinal plant.",
                    "predictions": []
                }
            )

        logger.info(f"Prediction successful: {predictions[0]['class_name']} ({predictions[0]['confidence']:.2%})")

        # Format response
        result = {
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None,
            "model_info": {
                "model_type": identifier.metadata.get('base_model', 'CNN'),
                "num_classes": identifier.num_classes,
                "confidence_threshold": identifier.confidence_threshold
            }
        }

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


@router.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    try:
        identifier = get_plant_identifier()
        return identifier.get_model_info()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )


@router.get("/model/classes")
async def get_model_classes():
    """Get list of plant classes the model can identify"""
    try:
        identifier = get_plant_identifier()
        return {
            "num_classes": identifier.num_classes,
            "classes": identifier.class_names
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting classes: {str(e)}"
        )


# Mock history endpoint (to be implemented with database)
@router.get("/history")
async def get_identification_history():
    """
    Get user's identification history
    TODO: Implement with database and user authentication
    """
    # This would typically fetch from database
    return {
        "message": "History endpoint - to be implemented with database",
        "history": []
    }


@router.post("/identify/detailed")
async def identify_plant_detailed(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Identify plant and return detailed information including
    medicinal properties from database

    Args:
        file: Uploaded image file
        top_k: Number of predictions

    Returns:
        Detailed identification with medicinal information
    """
    # Get basic identification
    identifier = get_plant_identifier()

    try:
        image_bytes = await file.read()
        predictions = identifier.predict_from_bytes(image_bytes, top_k=top_k)

        if not predictions:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "No confident predictions",
                    "predictions": []
                }
            )

        # TODO: Fetch plant details from database for each prediction
        # For now, return predictions with placeholder for database info

        detailed_predictions = []
        for pred in predictions:
            detailed_pred = {
                **pred,
                "plant_info": {
                    "message": "Database integration pending",
                    "local_name": pred['class_name'],
                    "scientific_name": f"{pred['class_name']} sp.",
                }
            }
            detailed_predictions.append(detailed_pred)

        return {
            "predictions": detailed_predictions,
            "top_prediction": detailed_predictions[0] if detailed_predictions else None
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
