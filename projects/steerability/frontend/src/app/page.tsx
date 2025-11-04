'use client'

import { useState } from 'react'

export default function HomePage() {
  const [prompt, setPrompt] = useState('')
  const [vector, setVector] = useState('positive_sentiment')
  const [result, setResult] = useState<any>(null)

  const handleSteer = async () => {
    const res = await fetch('http://localhost:8001/api/steering/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, vector_name: vector, strength: 1.0 })
    })
    const data = await res.json()
    setResult(data)
  }

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Control Panel</h2>

      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-xl font-semibold mb-4">Steering Controls</h3>

        <div className="space-y-4">
          <div>
            <label className="block mb-2 font-medium">Prompt</label>
            <textarea
              className="w-full p-3 border rounded"
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt..."
            />
          </div>

          <div>
            <label className="block mb-2 font-medium">Steering Vector</label>
            <select
              className="w-full p-3 border rounded"
              value={vector}
              onChange={(e) => setVector(e.target.value)}
            >
              <option value="positive_sentiment">Positive Sentiment</option>
              <option value="formal_tone">Formal Tone</option>
              <option value="concise">Concise</option>
            </select>
          </div>

          <button
            onClick={handleSteer}
            className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700"
          >
            Generate with Steering
          </button>
        </div>
      </div>

      {result && (
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow">
            <h4 className="font-semibold mb-2">Steered Output</h4>
            <p className="text-gray-700">{result.steered_output}</p>
            <div className="mt-4 text-sm">
              <span className="font-medium">Adherence:</span> {(result.adherence_score * 100).toFixed(1)}%
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow">
            <h4 className="font-semibold mb-2">Unsteered Output</h4>
            <p className="text-gray-700">{result.unsteered_output}</p>
          </div>
        </div>
      )}
    </div>
  )
}
