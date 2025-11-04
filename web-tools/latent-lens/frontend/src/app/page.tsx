'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'

interface SAEFeature {
  id: string
  description: string
  category?: string
  activation_examples: string[]
  statistics: {
    mean_activation: number
    max_activation: number
    frequency: number
  }
}

export default function HomePage() {
  const [features, setFeatures] = useState<SAEFeature[]>([])
  const [categories, setCategories] = useState<string[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    loadFeatures()
    loadCategories()
  }, [selectedCategory])

  const loadFeatures = async () => {
    setLoading(true)
    try {
      const url = selectedCategory
        ? `http://localhost:8002/api/features?category=${selectedCategory}`
        : 'http://localhost:8002/api/features'

      const response = await fetch(url)
      const data = await response.json()
      setFeatures(data)
    } catch (error) {
      console.error('Failed to load features:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadCategories = async () => {
    try {
      const response = await fetch('http://localhost:8002/api/categories')
      const data = await response.json()
      setCategories(data.categories)
    } catch (error) {
      console.error('Failed to load categories:', error)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      loadFeatures()
      return
    }

    try {
      const response = await fetch(`http://localhost:8002/api/search?q=${encodeURIComponent(searchQuery)}`)
      const data = await response.json()
      setFeatures(data.features)
    } catch (error) {
      console.error('Search failed:', error)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-purple-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            🔬 Latent Lens
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Explore interpretable features discovered in language models
          </p>
        </div>

        {/* Navigation */}
        <div className="flex justify-center gap-4 mb-8">
          <Link
            href="/"
            className="px-6 py-2 bg-indigo-600 text-white rounded-lg font-medium"
          >
            Gallery
          </Link>
          <Link
            href="/analyze"
            className="px-6 py-2 bg-white text-indigo-600 border-2 border-indigo-600 rounded-lg font-medium hover:bg-indigo-50"
          >
            Analyze Text
          </Link>
        </div>

        {/* Search & Filters */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="flex gap-4">
            <input
              type="text"
              placeholder="Search features..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
            />
            <button
              onClick={handleSearch}
              className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700"
            >
              Search
            </button>
          </div>

          {/* Category Filter */}
          <div className="mt-4 flex flex-wrap gap-2">
            <button
              onClick={() => {
                setSelectedCategory(null)
                setSearchQuery('')
              }}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedCategory === null
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              All
            </button>
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedCategory === category
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {category}
              </button>
            ))}
          </div>
        </div>

        {/* Feature Gallery */}
        {loading ? (
          <div className="text-center py-12">
            <div className="animate-spin text-6xl mb-4">🔬</div>
            <p className="text-gray-600">Loading features...</p>
          </div>
        ) : (
          <>
            <div className="mb-4 text-gray-600">
              Showing {features.length} features
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {features.map((feature) => (
                <FeatureCard key={feature.id} feature={feature} />
              ))}
            </div>
          </>
        )}

        {/* Footer */}
        <div className="mt-12 text-center text-gray-600 text-sm">
          <p>
            Powered by research from{' '}
            <a href="https://github.com/lfhvn/hidden-layer" className="text-indigo-600 hover:underline">
              Hidden Layer Lab
            </a>
          </p>
        </div>
      </div>
    </div>
  )
}

function FeatureCard({ feature }: { feature: SAEFeature }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow">
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="text-sm font-mono text-gray-500 mb-1">{feature.id}</div>
          <h3 className="text-lg font-semibold text-gray-900">{feature.description}</h3>
        </div>
        {feature.category && (
          <span className="px-2 py-1 bg-indigo-100 text-indigo-800 text-xs rounded-full">
            {feature.category}
          </span>
        )}
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="text-center p-2 bg-gray-50 rounded">
          <div className="text-xs text-gray-600">Mean</div>
          <div className="text-sm font-semibold">
            {(feature.statistics.mean_activation * 100).toFixed(0)}%
          </div>
        </div>
        <div className="text-center p-2 bg-gray-50 rounded">
          <div className="text-xs text-gray-600">Max</div>
          <div className="text-sm font-semibold">
            {(feature.statistics.max_activation * 100).toFixed(0)}%
          </div>
        </div>
        <div className="text-center p-2 bg-gray-50 rounded">
          <div className="text-xs text-gray-600">Freq</div>
          <div className="text-sm font-semibold">
            {(feature.statistics.frequency * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Activation Examples */}
      <div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-sm text-indigo-600 hover:text-indigo-800 font-medium mb-2"
        >
          {expanded ? '▼' : '▶'} {feature.activation_examples.length} examples
        </button>

        {expanded && (
          <div className="space-y-2 mt-2">
            {feature.activation_examples.map((example, i) => (
              <div key={i} className="text-sm text-gray-600 p-2 bg-gray-50 rounded">
                "{example}"
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
