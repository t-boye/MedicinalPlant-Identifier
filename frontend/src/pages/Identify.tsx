import { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Camera, Upload, X, Loader2 } from 'lucide-react';
import { plantService } from '../services/plantService';
import type { IdentificationResult } from '../types';
import PlantResult from '../components/plant/PlantResult';

export default function Identify() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [isIdentifying, setIsIdentifying] = useState(false);
  const [result, setResult] = useState<IdentificationResult | null>(null);
  const [error, setError] = useState<string>('');

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp'],
    },
    multiple: false,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setSelectedImage(file);
        setPreviewUrl(URL.createObjectURL(file));
        setResult(null);
        setError('');
      }
    },
  });

  const handleIdentify = async () => {
    if (!selectedImage) return;

    setIsIdentifying(true);
    setError('');

    try {
      const identificationResult = await plantService.identifyPlant(selectedImage);
      setResult(identificationResult);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to identify plant. Please try again.');
      console.error('Identification error:', err);
    } finally {
      setIsIdentifying(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setPreviewUrl('');
    setResult(null);
    setError('');
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Identify Medicinal Plant
        </h1>
        <p className="text-gray-600">
          Upload or capture a photo to identify the plant and learn about its medicinal properties
        </p>
      </div>

      {!result ? (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          {!selectedImage ? (
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-green-500 bg-green-50'
                  : 'border-gray-300 hover:border-green-400 hover:bg-gray-50'
              }`}
            >
              <input {...getInputProps()} />
              <div className="space-y-4">
                <div className="flex justify-center">
                  {isDragActive ? (
                    <Camera className="w-16 h-16 text-green-500" />
                  ) : (
                    <Upload className="w-16 h-16 text-gray-400" />
                  )}
                </div>
                <div>
                  <p className="text-lg font-medium text-gray-900">
                    {isDragActive ? 'Drop image here' : 'Upload or drop an image'}
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    PNG, JPG, JPEG or WEBP (max. 10MB)
                  </p>
                </div>
                <button className="mt-4 px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors">
                  Choose File
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative">
                <img
                  src={previewUrl}
                  alt="Selected plant"
                  className="w-full max-h-96 object-contain rounded-lg"
                />
                <button
                  onClick={handleReset}
                  className="absolute top-2 right-2 p-2 bg-white rounded-full shadow-md hover:bg-gray-100"
                >
                  <X className="w-5 h-5 text-gray-600" />
                </button>
              </div>

              {error && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                  {error}
                </div>
              )}

              <div className="flex gap-4">
                <button
                  onClick={handleIdentify}
                  disabled={isIdentifying}
                  className="flex-1 flex items-center justify-center px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  {isIdentifying ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Identifying...
                    </>
                  ) : (
                    <>
                      <Camera className="w-5 h-5 mr-2" />
                      Identify Plant
                    </>
                  )}
                </button>
                <button
                  onClick={handleReset}
                  className="px-6 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          <PlantResult result={result} />
          <button
            onClick={handleReset}
            className="w-full px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
          >
            Identify Another Plant
          </button>
        </div>
      )}
    </div>
  );
}
