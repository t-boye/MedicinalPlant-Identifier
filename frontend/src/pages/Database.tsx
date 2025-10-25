import { useState, useEffect } from 'react';
import { Search, Loader2 } from 'lucide-react';
import { plantService } from '../services/plantService';
import type { MedicinalPlant } from '../types';
import PlantCard from '../components/plant/PlantCard';

export default function Database() {
  const [plants, setPlants] = useState<MedicinalPlant[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    loadPlants();
  }, []);

  const loadPlants = async () => {
    setIsLoading(true);
    setError('');
    try {
      const response = await plantService.getAllPlants(100, 0);
      setPlants(response.plants);
    } catch (err: any) {
      setError('Failed to load plants. Please try again.');
      console.error('Error loading plants:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      loadPlants();
      return;
    }

    setIsLoading(true);
    setError('');
    try {
      const response = await plantService.searchPlants({ query: searchQuery });
      setPlants(response.plants);
    } catch (err: any) {
      setError('Search failed. Please try again.');
      console.error('Search error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Medicinal Plant Database
        </h1>
        <p className="text-gray-600">
          Browse and search our comprehensive collection of medicinal plants
        </p>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch} className="max-w-2xl mx-auto">
        <div className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by plant name, family, or medicinal use..."
            className="w-full px-4 py-3 pl-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
          />
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <button
            type="submit"
            className="absolute right-2 top-1/2 transform -translate-y-1/2 px-4 py-1.5 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors text-sm"
          >
            Search
          </button>
        </div>
      </form>

      {/* Error Message */}
      {error && (
        <div className="max-w-2xl mx-auto p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {error}
        </div>
      )}

      {/* Loading State */}
      {isLoading ? (
        <div className="flex justify-center items-center py-12">
          <Loader2 className="w-8 h-8 text-green-600 animate-spin" />
        </div>
      ) : plants.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-gray-500">No plants found</p>
        </div>
      ) : (
        <>
          <div className="text-sm text-gray-600">
            Found {plants.length} plant{plants.length !== 1 ? 's' : ''}
          </div>

          {/* Plants Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {plants.map((plant) => (
              <PlantCard key={plant.id} plant={plant} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
