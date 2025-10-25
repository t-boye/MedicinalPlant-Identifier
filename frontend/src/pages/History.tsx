import { useState, useEffect } from 'react';
import { Loader2, Clock, TrendingUp } from 'lucide-react';
import { plantService } from '../services/plantService';
import type { IdentificationHistory } from '../types';
import { Link } from 'react-router-dom';

export default function History() {
  const [history, setHistory] = useState<IdentificationHistory[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    setIsLoading(true);
    setError('');
    try {
      const data = await plantService.getHistory();
      setHistory(data);
    } catch (err: any) {
      setError('Failed to load identification history.');
      console.error('Error loading history:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-50';
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Identification History
        </h1>
        <p className="text-gray-600">
          View your past plant identifications
        </p>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {error}
        </div>
      )}

      {isLoading ? (
        <div className="flex justify-center items-center py-12">
          <Loader2 className="w-8 h-8 text-green-600 animate-spin" />
        </div>
      ) : history.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-lg shadow-sm border border-gray-200">
          <Clock className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <p className="text-gray-500 mb-4">No identification history yet</p>
          <Link
            to="/identify"
            className="inline-block px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
          >
            Identify Your First Plant
          </Link>
        </div>
      ) : (
        <div className="space-y-4">
          {history.map((item) => (
            <div
              key={item.id}
              className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow"
            >
              <div className="flex flex-col md:flex-row gap-6">
                {/* Image */}
                <div className="flex-shrink-0">
                  <img
                    src={item.image_url || '/placeholder-plant.png'}
                    alt={item.plant.local_name}
                    className="w-full md:w-32 h-32 object-cover rounded-lg"
                  />
                </div>

                {/* Info */}
                <div className="flex-1 space-y-2">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="text-xl font-semibold text-gray-900">
                        {item.plant.local_name}
                      </h3>
                      <p className="text-sm text-gray-500 italic">
                        {item.plant.scientific_name}
                      </p>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(item.confidence)}`}>
                      <div className="flex items-center gap-1">
                        <TrendingUp className="w-4 h-4" />
                        {Math.round(item.confidence * 100)}%
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center text-sm text-gray-500">
                    <Clock className="w-4 h-4 mr-1" />
                    {formatDate(item.identified_at)}
                  </div>

                  {item.plant.medicinal_uses && item.plant.medicinal_uses.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-3">
                      {item.plant.medicinal_uses.slice(0, 3).map((use, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-green-50 text-green-700 text-xs rounded-full"
                        >
                          {use}
                        </span>
                      ))}
                      {item.plant.medicinal_uses.length > 3 && (
                        <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">
                          +{item.plant.medicinal_uses.length - 3} more
                        </span>
                      )}
                    </div>
                  )}

                  <Link
                    to={`/plant/${item.plant_id}`}
                    className="inline-block text-green-600 hover:text-green-700 text-sm font-medium mt-2"
                  >
                    View Details →
                  </Link>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
