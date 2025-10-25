import { Link } from 'react-router-dom';
import type { MedicinalPlant } from '../../types';
import { Leaf, CheckCircle } from 'lucide-react';

interface PlantCardProps {
  plant: MedicinalPlant;
}

export default function PlantCard({ plant }: PlantCardProps) {
  return (
    <Link
      to={`/plant/${plant.id}`}
      className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow"
    >
      {/* Image */}
      <div className="relative h-48 bg-gray-100">
        {plant.image_urls && plant.image_urls.length > 0 ? (
          <img
            src={plant.image_urls[0]}
            alt={plant.local_name}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <Leaf className="w-16 h-16 text-gray-300" />
          </div>
        )}
        {plant.verified && (
          <div className="absolute top-2 right-2 bg-green-500 text-white p-1 rounded-full">
            <CheckCircle className="w-4 h-4" />
          </div>
        )}
      </div>

      {/* Content */}
      <div className="p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-1 line-clamp-1">
          {plant.local_name}
        </h3>
        <p className="text-sm text-gray-500 italic mb-3 line-clamp-1">
          {plant.scientific_name}
        </p>

        {plant.description && (
          <p className="text-sm text-gray-600 mb-3 line-clamp-2">
            {plant.description}
          </p>
        )}

        {/* Classification */}
        <div className="flex gap-2 text-xs text-gray-500 mb-3">
          {plant.family && (
            <span className="px-2 py-1 bg-gray-100 rounded">
              {plant.family}
            </span>
          )}
        </div>

        {/* Medicinal Uses */}
        {plant.medicinal_uses && plant.medicinal_uses.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {plant.medicinal_uses.slice(0, 2).map((use, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-green-50 text-green-700 text-xs rounded-full line-clamp-1"
              >
                {use}
              </span>
            ))}
            {plant.medicinal_uses.length > 2 && (
              <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">
                +{plant.medicinal_uses.length - 2}
              </span>
            )}
          </div>
        )}
      </div>
    </Link>
  );
}
