'use client'

import { useState } from 'react'
import { StrategySelector } from '@/components/StrategySelector'
import { StreamingDebateViewer } from '@/components/StreamingDebateViewer'
import { UsageIndicator } from '@/components/UsageIndicator'
import { useDebateStream } from '@/hooks/useDebateStream'

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
  const [useStreaming, setUseStreaming] = useState(true)

  const { messages, isStreaming, error, startStream, clearMessages } = useDebateStream()

  const handleStart = () => {
    if (!question.trim() || !strategy) return

    clearMessages()

    if (useStreaming) {
      // Use WebSocket streaming for real-time updates
      startStream(question, strategy.name, nAgents)
    }
  }

  // Get final result from messages
  const finalResult = messages.find(m => m.type === 'complete')

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
          <div className="mt-4 flex items-center justify-center gap-3">
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
              ⚡ Live Streaming
            </span>
            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
              🚀 Real-time Updates
            </span>
          </div>
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

              {error && (
                <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                  {error}
                </div>
              )}

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
                  disabled={isStreaming}
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
                  disabled={isStreaming}
                  maxLength={500}
                />
                <div className="text-xs text-gray-500 mt-1 text-right">
                  {question.length}/500
                </div>
              </div>

              {/* Streaming Toggle */}
              <div className="mb-6 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useStreaming}
                    onChange={(e) => setUseStreaming(e.target.checked)}
                    disabled={isStreaming}
                    className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm font-medium text-blue-900">
                    ⚡ Enable real-time streaming
                  </span>
                </label>
                <p className="text-xs text-blue-700 mt-1 ml-6">
                  See agents think as they work
                </p>
              </div>

              {/* Start Button */}
              <button
                onClick={handleStart}
                disabled={!question.trim() || !strategy || isStreaming}
                className={`w-full py-3 rounded-lg font-semibold text-white transition-colors ${
                  !question.trim() || !strategy || isStreaming
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                {isStreaming ? '🤖 Agents Thinking...' : '🚀 Start Arena'}
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
                    disabled={isStreaming}
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

              {messages.length === 0 && !isStreaming && (
                <div className="flex items-center justify-center h-96 text-gray-400">
                  <div className="text-center">
                    <div className="text-6xl mb-4">🤝</div>
                    <p className="text-lg">Select a strategy and ask a question to begin</p>
                    <p className="text-sm mt-2">Watch agents collaborate in real-time!</p>
                  </div>
                </div>
              )}

              {/* Streaming Messages */}
              <StreamingDebateViewer messages={messages} isStreaming={isStreaming} />

              {/* Action Buttons */}
              {finalResult && !isStreaming && (
                <div className="mt-6 flex gap-3">
                  <button
                    onClick={() => navigator.clipboard.writeText(finalResult.content)}
                    className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium transition-colors"
                  >
                    📋 Copy Result
                  </button>
                  <button
                    onClick={() => {
                      const text = `Question: ${question}\n\nStrategy: ${strategy?.display_name}\n\nResult:\n${finalResult.content}`
                      navigator.clipboard.writeText(text)
                    }}
                    className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium transition-colors"
                  >
                    📤 Share
                  </button>
                  <button
                    onClick={clearMessages}
                    className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium transition-colors"
                  >
                    🔄 New Question
                  </button>
                </div>
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
