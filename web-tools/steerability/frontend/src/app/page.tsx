'use client'

import { useState } from 'react'
import { UsageIndicator } from '@/components/UsageIndicator'

export default function HomePage() {
  const [prompt, setPrompt] = useState('')
  const [vector, setVector] = useState('positive_sentiment')
  const [strength, setStrength] = useState(1.0)
  const [result, setResult] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSteer = async () => {
    if (!prompt.trim()) return

    setIsLoading(true)
    setError(null)

    try {
      const res = await fetch('http://localhost:8000/api/steering/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, vector_name: vector, strength })
      })

      if (res.status === 429) {
        const data = await res.json()
        setError(`Rate limit exceeded. Try again in ${Math.ceil(data.detail.retry_after / 60)} minutes.`)
        setIsLoading(false)
        return
      }

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      console.error('Error:', err)
      setError('An error occurred. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            🎛️ Steerability Dashboard
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Live LLM steering with real-time adherence metrics
          </p>
        </div>

        {/* Usage Indicator */}
        <UsageIndicator />

        {/* Controls */}
        <div className="max-w-4xl mx-auto">
          <div className="bg-white p-8 rounded-xl shadow-lg mb-6">
            <h2 className="text-2xl font-bold mb-6">Steering Controls</h2>

            {error && (
              <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                {error}
              </div>
            )}

            <div className="space-y-6">
              {/* Prompt Input */}
              <div>
                <label className="block mb-2 font-medium text-gray-700">Prompt</label>
                <textarea
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  rows={4}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Enter your prompt... (e.g., 'Write about the weather')"
                  disabled={isLoading}
                />
              </div>

              {/* Vector Selection */}
              <div>
                <label className="block mb-2 font-medium text-gray-700">Steering Vector</label>
                <select
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                  value={vector}
                  onChange={(e) => setVector(e.target.value)}
                  disabled={isLoading}
                >
                  <option value="positive_sentiment">✨ Positive Sentiment</option>
                  <option value="formal_tone">🎩 Formal Tone</option>
                  <option value="concise">📝 Concise</option>
                </select>
              </div>

              {/* Strength Slider */}
              <div>
                <label className="block mb-2 font-medium text-gray-700">
                  Steering Strength: {strength.toFixed(1)}x
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="3.0"
                  step="0.1"
                  value={strength}
                  onChange={(e) => setStrength(parseFloat(e.target.value))}
                  className="w-full"
                  disabled={isLoading}
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Subtle</span>
                  <span>Strong</span>
                  <span>Very Strong</span>
                </div>
              </div>

              {/* Generate Button */}
              <button
                onClick={handleSteer}
                disabled={!prompt.trim() || isLoading}
                className={`w-full py-3 rounded-lg font-semibold text-white transition-colors ${
                  !prompt.trim() || isLoading
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-purple-600 hover:bg-purple-700'
                }`}
              >
                {isLoading ? '⚙️ Generating...' : '🚀 Generate with Steering'}
              </button>
            </div>
          </div>

          {/* Results */}
          {result && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Steered Output */}
              <div className="bg-white p-6 rounded-xl shadow-lg border-2 border-purple-200">
                <h3 className="font-semibold text-lg mb-3 flex items-center">
                  <span className="mr-2">🎯</span> Steered Output
                </h3>
                <p className="text-gray-800 whitespace-pre-wrap mb-4">{result.steered_output}</p>
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium text-gray-700">Adherence Score:</span>
                    <span className="font-bold text-purple-600">
                      {(result.adherence_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  {result.constraints_satisfied && (
                    <div className="mt-2 text-xs text-green-600 flex items-center">
                      <span className="mr-1">✓</span> Constraints satisfied
                    </div>
                  )}
                </div>
              </div>

              {/* Unsteered Output */}
              <div className="bg-white p-6 rounded-xl shadow-lg border-2 border-gray-200">
                <h3 className="font-semibold text-lg mb-3 flex items-center">
                  <span className="mr-2">📄</span> Unsteered Output
                </h3>
                <p className="text-gray-800 whitespace-pre-wrap">{result.unsteered_output}</p>
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="text-sm text-gray-500">
                    Original output without steering
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Info Box */}
          <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
            <h3 className="font-semibold text-blue-900 mb-2">How it works</h3>
            <p className="text-sm text-blue-800 mb-3">
              Steerability uses activation vectors to guide LLM outputs toward desired characteristics
              without fine-tuning. Try different vectors and strengths to see the effect!
            </p>
            <p className="text-xs text-blue-600">
              Free tier: 5 requests per hour •
              <a href="#" className="underline ml-1">Bring your own API key</a> for unlimited access
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-gray-600 text-sm">
          <p>
            Powered by research from{' '}
            <a href="https://github.com/lfhvn/hidden-layer" className="text-purple-600 hover:underline">
              Hidden Layer Lab
            </a>
          </p>
        </div>
      </div>
    </div>
  )
}
