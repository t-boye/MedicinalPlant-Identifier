from sqlalchemy import Column, Integer, String, Text, JSON, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class MedicinalPlant(Base):
    __tablename__ = "medicinal_plants"

    id = Column(Integer, primary_key=True, index=True)

    # Names
    local_name = Column(String, index=True, nullable=False)
    scientific_name = Column(String, index=True, nullable=False)
    common_names = Column(JSON)  # Store multiple common names

    # Classification
    family = Column(String)
    genus = Column(String)
    species = Column(String)

    # Physical Description
    description = Column(Text)
    habitat = Column(Text)
    distribution = Column(Text)

    # Phytochemical Data
    active_compounds = Column(JSON)  # List of phytochemicals
    chemical_composition = Column(JSON)  # Detailed composition data

    # Therapeutic Properties
    medicinal_uses = Column(JSON)  # List of therapeutic uses
    preparation_methods = Column(JSON)  # How to prepare
    dosage_info = Column(Text)
    contraindications = Column(Text)
    side_effects = Column(Text)

    # Additional Information
    traditional_knowledge = Column(Text)
    research_references = Column(JSON)  # Academic references

    # Image Data
    image_urls = Column(JSON)  # Multiple plant images

    # Metadata
    confidence_score = Column(Float)
    verified = Column(Integer, default=0)  # 0: unverified, 1: verified
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
