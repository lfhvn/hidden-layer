'use client'

import { useState } from 'react'
import { StrategySelector } from '@/components/StrategySelector'
import { DebateViewer } from '@/components/DebateViewer'
import { UsageIndicator } from '@/components/UsageIndicator'

interface Strategy {
  name: string
  display_name: string
  description: string
  best_for: string
  typical_agents: number
}

export default function ArenaPage() {
  const [question, setQuestion] = useState('')
  const [strategy, setStrategy] = useState<Strategy | null>(null)
  const [nAgents, setNAgents] = useState(3)
  const [isRunning, setIsRunning] = useState(false)
  const [messages, setMessages] = useState<any[]>([])
  const [result, setResult] = useState<string | null>(null)

  const handleStart = async () => {
    if (!question.trim() || !strategy) return

    setIsRunning(true)
    setMessages([])
    setResult(null)

    // Use REST API endpoint (simpler for MVP)
    // TODO: Implement WebSocket streaming for real-time updates
    try {
      const response = await fetch('http://localhost:8000/api/debate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          strategy: strategy.name,
          n_agents: nAgents
        })
      })

      if (response.status === 429) {
        const data = await response.json()
        alert(`Rate limit exceeded. Try again in ${Math.ceil(data.detail.retry_after / 60)} minutes.`)
        setIsRunning(false)
        return
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const data = await response.json()
      setResult(data.result)
    } catch (error) {
      console.error('Error:', error)
      alert('An error occurred. Please try again.')
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            🤝 Multi-Agent Arena
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Watch AI agents debate, critique, and solve problems in real-time
          </p>
        </div>

        {/* Usage Indicator */}
        <div className="mb-8">
          <UsageIndicator />
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left: Controls */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold mb-6">Setup</h2>

              {/* Strategy Selector */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Strategy
                </label>
                <StrategySelector
                  onSelect={setStrategy}
                  selected={strategy}
                />
              </div>

              {/* Number of Agents */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Agents: {nAgents}
                </label>
                <input
                  type="range"
                  min="2"
                  max="5"
                  value={nAgents}
                  onChange={(e) => setNAgents(parseInt(e.target.value))}
                  className="w-full"
                  disabled={isRunning}
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>2</span>
                  <span>3</span>
                  <span>4</span>
                  <span>5</span>
                </div>
              </div>

              {/* Question Input */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Your Question
                </label>
                <textarea
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows={4}
                  placeholder="e.g., Should we invest in renewable energy?"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  disabled={isRunning}
                  maxLength={500}
                />
                <div className="text-xs text-gray-500 mt-1 text-right">
                  {question.length}/500
                </div>
              </div>

              {/* Start Button */}
              <button
                onClick={handleStart}
                disabled={!question.trim() || !strategy || isRunning}
                className={`w-full py-3 rounded-lg font-semibold text-white transition-colors ${
                  !question.trim() || !strategy || isRunning
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                {isRunning ? '🤖 Agents Thinking...' : '🚀 Start Arena'}
              </button>

              {/* Info */}
              {strategy && (
                <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                  <h3 className="font-semibold text-blue-900 mb-2">
                    {strategy.display_name}
                  </h3>
                  <p className="text-sm text-blue-700 mb-2">
                    {strategy.description}
                  </p>
                  <p className="text-xs text-blue-600">
                    <strong>Best for:</strong> {strategy.best_for}
                  </p>
                </div>
              )}
            </div>

            {/* Example Questions */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="font-semibold mb-3">Example Questions</h3>
              <div className="space-y-2">
                {[
                  "Should AI be regulated by governments?",
                  "What's the best approach to teaching kids about AI?",
                  "How can we reduce urban traffic congestion?"
                ].map((example, i) => (
                  <button
                    key={i}
                    onClick={() => setQuestion(example)}
                    disabled={isRunning}
                    className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded transition-colors disabled:opacity-50"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right: Results */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-lg p-6 min-h-[600px]">
              <h2 className="text-2xl font-bold mb-6">Arena</h2>

              {!result && !isRunning && (
                <div className="flex items-center justify-center h-96 text-gray-400">
                  <div className="text-center">
                    <div className="text-6xl mb-4">🤝</div>
                    <p className="text-lg">Select a strategy and ask a question to begin</p>
                  </div>
                </div>
              )}

              {isRunning && (
                <div className="flex items-center justify-center h-96">
                  <div className="text-center">
                    <div className="animate-spin text-6xl mb-4">🤖</div>
                    <p className="text-lg text-gray-600">Agents are thinking...</p>
                    <p className="text-sm text-gray-400 mt-2">This may take 30-60 seconds</p>
                  </div>
                </div>
              )}

              {result && (
                <DebateViewer
                  question={question}
                  strategy={strategy!}
                  result={result}
                  nAgents={nAgents}
                />
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-gray-600 text-sm">
          <p>
            Powered by research from <a href="https://github.com/lfhvn/hidden-layer" className="text-blue-600 hover:underline">Hidden Layer Lab</a>
          </p>
          <p className="mt-2">
            Free tier: 3 debates per hour •
            <a href="#" className="text-blue-600 hover:underline ml-1">Bring your own API key</a> for unlimited access
          </p>
        </div>
      </div>
    </div>
  )
}
