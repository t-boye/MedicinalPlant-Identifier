import { Link } from 'react-router-dom';
import { Camera, Database, Leaf, Search, BookOpen, Shield } from 'lucide-react';

export default function Home() {
  const features = [
    {
      icon: Camera,
      title: 'Identify Plants',
      description: 'Upload or capture a photo to instantly identify medicinal plants',
      link: '/identify',
      color: 'green',
    },
    {
      icon: Database,
      title: 'Plant Database',
      description: 'Browse our comprehensive database of medicinal plants',
      link: '/database',
      color: 'blue',
    },
    {
      icon: BookOpen,
      title: 'Detailed Information',
      description: 'Learn about phytochemicals, uses, and preparation methods',
      link: '/database',
      color: 'purple',
    },
    {
      icon: Shield,
      title: 'Safety First',
      description: 'Get information about contraindications and side effects',
      link: '/database',
      color: 'red',
    },
  ];

  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="text-center space-y-6 py-12">
        <div className="flex justify-center">
          <Leaf className="w-20 h-20 text-green-600" />
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-gray-900">
          Medicinal Plant Identifier
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Discover the healing power of nature. Identify medicinal plants and learn about their therapeutic properties.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
          <Link
            to="/identify"
            className="inline-flex items-center justify-center px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium"
          >
            <Camera className="w-5 h-5 mr-2" />
            Start Identifying
          </Link>
          <Link
            to="/database"
            className="inline-flex items-center justify-center px-6 py-3 bg-white text-green-600 border-2 border-green-600 rounded-lg hover:bg-green-50 transition-colors font-medium"
          >
            <Search className="w-5 h-5 mr-2" />
            Browse Database
          </Link>
        </div>
      </section>

      {/* Features Grid */}
      <section className="grid md:grid-cols-2 gap-6">
        {features.map((feature) => {
          const Icon = feature.icon;
          const colorClasses = {
            green: 'bg-green-100 text-green-600',
            blue: 'bg-blue-100 text-blue-600',
            purple: 'bg-purple-100 text-purple-600',
            red: 'bg-red-100 text-red-600',
          };

          return (
            <Link
              key={feature.title}
              to={feature.link}
              className="p-6 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow border border-gray-200"
            >
              <div className={`w-12 h-12 rounded-lg ${colorClasses[feature.color as keyof typeof colorClasses]} flex items-center justify-center mb-4`}>
                <Icon className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-600">
                {feature.description}
              </p>
            </Link>
          );
        })}
      </section>

      {/* Info Section */}
      <section className="bg-white rounded-lg shadow-sm p-8 border border-gray-200">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          How It Works
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="w-12 h-12 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-3 font-bold text-lg">
              1
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Capture or Upload</h3>
            <p className="text-gray-600 text-sm">
              Take a photo or upload an image of the medicinal plant
            </p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-3 font-bold text-lg">
              2
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">AI Identification</h3>
            <p className="text-gray-600 text-sm">
              Our AI analyzes the image and identifies the plant species
            </p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-3 font-bold text-lg">
              3
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Learn & Discover</h3>
            <p className="text-gray-600 text-sm">
              Get detailed information about medicinal properties and uses
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
