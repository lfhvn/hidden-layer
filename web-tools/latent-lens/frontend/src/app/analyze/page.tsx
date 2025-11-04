'use client'

import { useState } from 'react'
import Link from 'next/link'

interface FeatureActivation {
  feature_id: string
  feature_description: string
  activation_strength: number
  relevant_spans: string[]
}

export default function AnalyzePage() {
  const [text, setText] = useState('')
  const [activations, setActivations] = useState<FeatureActivation[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const handleAnalyze = async () => {
    if (!text.trim()) return

    setIsAnalyzing(true)

    try {
      const response = await fetch('http://localhost:8002/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, top_k: 10 })
      })

      const data = await response.json()
      setActivations(data)
    } catch (error) {
      console.error('Analysis failed:', error)
    } finally {
      setIsAnalyzing(false)
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
            Analyze your text to see which features activate
          </p>
        </div>

        {/* Navigation */}
        <div className="flex justify-center gap-4 mb-8">
          <Link
            href="/"
            className="px-6 py-2 bg-white text-indigo-600 border-2 border-indigo-600 rounded-lg font-medium hover:bg-indigo-50"
          >
            Gallery
          </Link>
          <Link
            href="/analyze"
            className="px-6 py-2 bg-indigo-600 text-white rounded-lg font-medium"
          >
            Analyze Text
          </Link>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold mb-4">Your Text</h2>

            <textarea
              className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              rows={12}
              placeholder="Paste your text here to analyze which SAE features activate..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={isAnalyzing}
            />

            <div className="flex justify-between items-center mt-4">
              <div className="text-sm text-gray-600">
                {text.length} characters
              </div>

              <button
                onClick={handleAnalyze}
                disabled={!text.trim() || isAnalyzing}
                className={`px-6 py-3 rounded-lg font-semibold text-white transition-colors ${
                  !text.trim() || isAnalyzing
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-indigo-600 hover:bg-indigo-700'
                }`}
              >
                {isAnalyzing ? '⚙️ Analyzing...' : '🔍 Analyze'}
              </button>
            </div>

            {/* Sample Texts */}
            <div className="mt-6">
              <h3 className="font-semibold text-gray-700 mb-2">Try these examples:</h3>
              <div className="space-y-2">
                {[
                  "I absolutely love exploring new cities! Paris and Tokyo are my favorites.",
                  "The function returns a boolean value when the API request completes successfully.",
                  "What do you think about the new features? How should we proceed?"
                ].map((example, i) => (
                  <button
                    key={i}
                    onClick={() => setText(example)}
                    disabled={isAnalyzing}
                    className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded transition-colors disabled:opacity-50"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Results */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold mb-4">Activated Features</h2>

            {activations.length === 0 && !isAnalyzing && (
              <div className="flex items-center justify-center h-96 text-gray-400">
                <div className="text-center">
                  <div className="text-6xl mb-4">📊</div>
                  <p className="text-lg">Enter text to see feature activations</p>
                </div>
              </div>
            )}

            {isAnalyzing && (
              <div className="flex items-center justify-center h-96">
                <div className="text-center">
                  <div className="animate-spin text-6xl mb-4">⚙️</div>
                  <p className="text-gray-600">Analyzing features...</p>
                </div>
              </div>
            )}

            {activations.length > 0 && !isAnalyzing && (
              <div className="space-y-4">
                {activations.map((activation, i) => (
                  <div key={i} className="border-l-4 border-indigo-500 bg-indigo-50 p-4 rounded">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex-1">
                        <div className="text-sm font-mono text-gray-500 mb-1">
                          {activation.feature_id}
                        </div>
                        <div className="font-semibold text-gray-900">
                          {activation.feature_description}
                        </div>
                      </div>
                      <div className="ml-4">
                        <div className="text-sm text-gray-600">Strength</div>
                        <div className="text-2xl font-bold text-indigo-600">
                          {(activation.activation_strength * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>

                    {/* Activation Bar */}
                    <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                      <div
                        className="bg-indigo-600 h-2 rounded-full transition-all"
                        style={{ width: `${activation.activation_strength * 100}%` }}
                      />
                    </div>
                  </div>
                ))}

                <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <p className="text-sm text-yellow-800">
                    <strong>Note:</strong> This is a demonstration using simplified pattern matching.
                    A real implementation would use the actual SAE model to compute activations.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Info */}
        <div className="mt-8 bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-bold mb-4">About Feature Analysis</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-indigo-900 mb-2">What are SAE Features?</h4>
              <p className="text-sm text-gray-700">
                Sparse Autoencoders (SAEs) discover interpretable features that language models use internally.
                Each feature represents a pattern or concept like "cities", "sentiment", or "technical jargon".
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-indigo-900 mb-2">How does this work?</h4>
              <p className="text-sm text-gray-700">
                When you analyze text, we compute how strongly each discovered feature activates.
                Higher activation means the feature is more relevant to your text.
                This helps understand what patterns the model detects.
              </p>
            </div>
          </div>
        </div>

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
