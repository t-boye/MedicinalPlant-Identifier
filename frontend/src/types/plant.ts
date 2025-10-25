export interface MedicinalPlant {
  id: number;

  // Names
  local_name: string;
  scientific_name: string;
  common_names: string[];

  // Classification
  family: string;
  genus: string;
  species: string;

  // Physical Description
  description: string;
  habitat: string;
  distribution: string;

  // Phytochemical Data
  active_compounds: string[];
  chemical_composition: Record<string, any>;

  // Therapeutic Properties
  medicinal_uses: string[];
  preparation_methods: string[];
  dosage_info: string;
  contraindications: string;
  side_effects: string;

  // Additional Information
  traditional_knowledge: string;
  research_references: ResearchReference[];

  // Image Data
  image_urls: string[];

  // Metadata
  confidence_score: number;
  verified: boolean;
  created_at: string;
  updated_at: string;
}

export interface ResearchReference {
  title: string;
  authors?: string[];
  journal?: string;
  year?: number;
  url?: string;
}

export interface IdentificationResult {
  plant: MedicinalPlant;
  confidence: number;
  matches: MedicinalPlant[];
}

export interface IdentificationHistory {
  id: number;
  plant_id: number;
  plant: MedicinalPlant;
  image_url: string;
  confidence: number;
  identified_at: string;
}

export interface PlantSearchParams {
  query?: string;
  family?: string;
  medicinal_use?: string;
  limit?: number;
  offset?: number;
}

export interface PlantSearchResponse {
  plants: MedicinalPlant[];
  total: number;
  offset: number;
  limit: number;
}
