'use client'

import { StreamMessage } from '@/hooks/useDebateStream'

interface StreamingDebateViewerProps {
  messages: StreamMessage[]
  isStreaming: boolean
}

export function StreamingDebateViewer({ messages, isStreaming }: StreamingDebateViewerProps) {
  if (messages.length === 0) {
    return null
  }

  return (
    <div className="space-y-4">
      {messages.map((message, index) => (
        <MessageBubble key={index} message={message} />
      ))}

      {isStreaming && (
        <div className="flex items-center gap-2 text-blue-600">
          <div className="animate-spin text-2xl">⚙️</div>
          <span className="text-sm font-medium">Agents thinking...</span>
        </div>
      )}
    </div>
  )
}

function MessageBubble({ message }: { message: StreamMessage }) {
  const getIcon = () => {
    switch (message.type) {
      case 'status': return '📋'
      case 'agent': return '🤖'
      case 'judge': return '⚖️'
      case 'synthesis': return '🔄'
      case 'complete': return '✅'
      case 'error': return '❌'
      default: return '💬'
    }
  }

  const getBackgroundColor = () => {
    switch (message.type) {
      case 'status': return 'bg-blue-50 border-blue-200'
      case 'agent': return 'bg-green-50 border-green-200'
      case 'judge': return 'bg-purple-50 border-purple-200'
      case 'synthesis': return 'bg-yellow-50 border-yellow-200'
      case 'complete': return 'bg-gray-50 border-gray-300'
      case 'error': return 'bg-red-50 border-red-200'
      default: return 'bg-gray-50 border-gray-200'
    }
  }

  const getTextColor = () => {
    switch (message.type) {
      case 'status': return 'text-blue-800'
      case 'agent': return 'text-green-800'
      case 'judge': return 'text-purple-800'
      case 'synthesis': return 'text-yellow-800'
      case 'complete': return 'text-gray-800'
      case 'error': return 'text-red-800'
      default: return 'text-gray-800'
    }
  }

  // Special rendering for complete message
  if (message.type === 'complete') {
    return (
      <div className="bg-white border-2 border-green-300 rounded-xl shadow-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <span className="text-3xl">{getIcon()}</span>
          <h3 className="text-xl font-bold text-gray-900">Final Result</h3>
        </div>
        <div className="prose max-w-none">
          <p className="text-gray-800 whitespace-pre-wrap leading-relaxed">
            {message.content}
          </p>
        </div>
        {message.metadata && (
          <div className="mt-4 pt-4 border-t border-gray-200 text-xs text-gray-500">
            Strategy: {message.metadata.strategy} • Agents: {message.metadata.n_agents}
          </div>
        )}
      </div>
    )
  }

  // Special rendering for agent messages
  if (message.type === 'agent') {
    return (
      <div className="bg-white border-l-4 border-green-500 rounded-lg shadow p-4">
        <div className="flex items-start gap-3">
          <span className="text-2xl">{getIcon()}</span>
          <div className="flex-1">
            <div className="font-semibold text-green-900 mb-1">
              {message.role || `Agent ${message.agent_id}`}
            </div>
            <p className="text-gray-700 leading-relaxed">{message.content}</p>
          </div>
        </div>
      </div>
    )
  }

  // Standard rendering for status messages
  return (
    <div className={`border rounded-lg p-3 ${getBackgroundColor()}`}>
      <div className="flex items-center gap-2">
        <span className="text-xl">{getIcon()}</span>
        <span className={`text-sm font-medium ${getTextColor()}`}>
          {message.content}
        </span>
      </div>
    </div>
  )
}
