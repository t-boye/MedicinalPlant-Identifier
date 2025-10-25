import type { IdentificationResult } from '../../types';
import { CheckCircle, AlertTriangle, Pill, Leaf, FlaskConical } from 'lucide-react';

interface PlantResultProps {
  result: IdentificationResult;
}

export default function PlantResult({ result }: PlantResultProps) {
  const { plant, confidence } = result;

  const getConfidenceLevel = () => {
    if (confidence >= 0.8) return { text: 'High', color: 'green' };
    if (confidence >= 0.6) return { text: 'Medium', color: 'yellow' };
    return { text: 'Low', color: 'red' };
  };

  const confidenceLevel = getConfidenceLevel();

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      {/* Header with confidence */}
      <div className={`p-4 bg-${confidenceLevel.color}-50 border-b border-${confidenceLevel.color}-200`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <CheckCircle className={`w-6 h-6 text-${confidenceLevel.color}-600`} />
            <span className={`font-semibold text-${confidenceLevel.color}-900`}>
              Identification Complete
            </span>
          </div>
          <div className={`px-3 py-1 bg-${confidenceLevel.color}-100 text-${confidenceLevel.color}-700 rounded-full text-sm font-medium`}>
            {Math.round(confidence * 100)}% Confidence
          </div>
        </div>
      </div>

      {/* Plant Details */}
      <div className="p-6 space-y-6">
        {/* Names */}
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-1">
            {plant.local_name}
          </h2>
          <p className="text-lg text-gray-600 italic mb-2">
            {plant.scientific_name}
          </p>
          {plant.common_names && plant.common_names.length > 0 && (
            <p className="text-sm text-gray-500">
              Also known as: {plant.common_names.join(', ')}
            </p>
          )}
        </div>

        {/* Classification */}
        <div className="grid grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
          <div>
            <p className="text-xs text-gray-500 mb-1">Family</p>
            <p className="font-medium text-gray-900">{plant.family || 'N/A'}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Genus</p>
            <p className="font-medium text-gray-900">{plant.genus || 'N/A'}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Species</p>
            <p className="font-medium text-gray-900">{plant.species || 'N/A'}</p>
          </div>
        </div>

        {/* Description */}
        {plant.description && (
          <div>
            <h3 className="flex items-center text-lg font-semibold text-gray-900 mb-2">
              <Leaf className="w-5 h-5 mr-2 text-green-600" />
              Description
            </h3>
            <p className="text-gray-700">{plant.description}</p>
          </div>
        )}

        {/* Medicinal Uses */}
        {plant.medicinal_uses && plant.medicinal_uses.length > 0 && (
          <div>
            <h3 className="flex items-center text-lg font-semibold text-gray-900 mb-3">
              <Pill className="w-5 h-5 mr-2 text-green-600" />
              Medicinal Uses
            </h3>
            <ul className="space-y-2">
              {plant.medicinal_uses.map((use, index) => (
                <li key={index} className="flex items-start">
                  <span className="w-1.5 h-1.5 bg-green-600 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                  <span className="text-gray-700">{use}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Active Compounds */}
        {plant.active_compounds && plant.active_compounds.length > 0 && (
          <div>
            <h3 className="flex items-center text-lg font-semibold text-gray-900 mb-3">
              <FlaskConical className="w-5 h-5 mr-2 text-green-600" />
              Active Compounds
            </h3>
            <div className="flex flex-wrap gap-2">
              {plant.active_compounds.map((compound, index) => (
                <span
                  key={index}
                  className="px-3 py-1 bg-blue-50 text-blue-700 text-sm rounded-full"
                >
                  {compound}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Safety Information */}
        {(plant.contraindications || plant.side_effects) && (
          <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
            <div className="flex items-start space-x-2">
              <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h4 className="font-semibold text-amber-900 mb-2">Safety Information</h4>
                {plant.contraindications && (
                  <div className="mb-2">
                    <p className="text-sm font-medium text-amber-800">Contraindications:</p>
                    <p className="text-sm text-amber-700">{plant.contraindications}</p>
                  </div>
                )}
                {plant.side_effects && (
                  <div>
                    <p className="text-sm font-medium text-amber-800">Side Effects:</p>
                    <p className="text-sm text-amber-700">{plant.side_effects}</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Dosage Info */}
        {plant.dosage_info && (
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Dosage Information
            </h3>
            <p className="text-gray-700">{plant.dosage_info}</p>
          </div>
        )}
      </div>
    </div>
  );
}
