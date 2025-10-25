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
    formData.append('image', imageFile);

    const response = await apiClient.post('/api/recognition/identify', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
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
