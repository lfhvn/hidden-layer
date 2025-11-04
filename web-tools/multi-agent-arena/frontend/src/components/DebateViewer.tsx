'use client'

interface Strategy {
  name: string
  display_name: string
  description: string
  best_for: string
  typical_agents: number
}

interface DebateViewerProps {
  question: string
  strategy: Strategy
  result: string
  nAgents: number
}

export function DebateViewer({ question, strategy, result, nAgents }: DebateViewerProps) {
  return (
    <div className="space-y-6">
      {/* Question */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg border-l-4 border-blue-500">
        <div className="text-sm font-semibold text-blue-900 mb-2">QUESTION</div>
        <div className="text-lg font-medium text-gray-900">{question}</div>
        <div className="text-sm text-gray-600 mt-2">
          Strategy: {strategy.display_name} • Agents: {nAgents}
        </div>
      </div>

      {/* Result */}
      <div className="prose max-w-none">
        <div className="text-sm font-semibold text-gray-700 mb-3">RESULT</div>
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <div className="whitespace-pre-wrap text-gray-800 leading-relaxed">
            {result}
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={() => navigator.clipboard.writeText(result)}
          className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium transition-colors"
        >
          📋 Copy Result
        </button>
        <button
          onClick={() => {
            const text = `Question: ${question}\n\nStrategy: ${strategy.display_name}\n\nResult:\n${result}`
            navigator.clipboard.writeText(text)
          }}
          className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium transition-colors"
        >
          📤 Share
        </button>
      </div>

      {/* Info */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <div className="text-sm text-yellow-800">
          <strong>💡 Tip:</strong> Try running the same question with different strategies to compare results!
        </div>
      </div>
    </div>
  )
}
