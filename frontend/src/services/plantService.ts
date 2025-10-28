import { apiClient } from './api';
import type {
  MedicinalPlant,
  IdentificationResult,
  IdentificationHistory,
  PlantSearchParams,
  PlantSearchResponse,
} from '../types';

export const plantService = {
  // Search plants
  searchPlants: async (params: PlantSearchParams): Promise<PlantSearchResponse> => {
    const response = await apiClient.get('/api/plants/search', { params });
    return response.data;
  },

  // Get all plants
  getAllPlants: async (limit = 50, offset = 0): Promise<PlantSearchResponse> => {
    const response = await apiClient.get('/api/plants', {
      params: { limit, offset },
    });
    return response.data;
  },

  // Get plant by ID
  getPlantById: async (id: number): Promise<MedicinalPlant> => {
    const response = await apiClient.get(`/api/plants/${id}`);
    return response.data;
  },

  // Identify plant from image
  identifyPlant: async (imageFile: File): Promise<IdentificationResult> => {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await apiClient.post('/api/recognition/identify', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    // Transform backend response to match frontend types
    const backendData = response.data;
    const topPrediction = backendData.top_prediction || backendData.predictions[0];

    if (!topPrediction) {
      throw new Error('No predictions returned from the model');
    }

    // Create a mock plant object from the prediction
    // TODO: Replace with actual database lookup once implemented
    const plant: MedicinalPlant = {
      id: topPrediction.class_id || 0,
      local_name: topPrediction.class_name.replace(/_/g, ' '),
      scientific_name: `${topPrediction.class_name.replace(/_/g, ' ')} sp.`,
      common_names: [topPrediction.class_name.replace(/_/g, ' ')],
      family: 'Medicinal Plant Family',
      genus: topPrediction.class_name.split('_')[0],
      species: 'species',
      description: `This is ${topPrediction.class_name.replace(/_/g, ' ')}, a medicinal plant identified by our AI model.`,
      habitat: 'Information pending database integration',
      distribution: 'Information pending database integration',
      active_compounds: [],
      chemical_composition: {},
      medicinal_uses: ['Traditional medicinal uses information pending database integration'],
      preparation_methods: [],
      dosage_info: 'Consult with a healthcare professional for proper dosage',
      contraindications: 'Contraindications information pending database integration',
      side_effects: 'Side effects information pending database integration',
      traditional_knowledge: '',
      research_references: [],
      image_urls: [],
      confidence_score: topPrediction.confidence,
      verified: false,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    // Create matches array from other predictions
    const matches: MedicinalPlant[] = backendData.predictions?.slice(1).map((pred: any) => ({
      id: pred.class_id || 0,
      local_name: pred.class_name.replace(/_/g, ' '),
      scientific_name: `${pred.class_name.replace(/_/g, ' ')} sp.`,
      common_names: [pred.class_name.replace(/_/g, ' ')],
      family: 'Medicinal Plant Family',
      genus: pred.class_name.split('_')[0],
      species: 'species',
      description: `Possible match: ${pred.class_name.replace(/_/g, ' ')}`,
      habitat: '',
      distribution: '',
      active_compounds: [],
      chemical_composition: {},
      medicinal_uses: [],
      preparation_methods: [],
      dosage_info: '',
      contraindications: '',
      side_effects: '',
      traditional_knowledge: '',
      research_references: [],
      image_urls: [],
      confidence_score: pred.confidence,
      verified: false,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    })) || [];

    return {
      plant,
      confidence: topPrediction.confidence,
      matches,
    };
  },

  // Get identification history
  getHistory: async (): Promise<IdentificationHistory[]> => {
    const response = await apiClient.get('/api/recognition/history');
    return response.data;
  },

  // Get featured/popular plants
  getFeaturedPlants: async (): Promise<MedicinalPlant[]> => {
    const response = await apiClient.get('/api/plants/featured');
    return response.data;
  },
};
