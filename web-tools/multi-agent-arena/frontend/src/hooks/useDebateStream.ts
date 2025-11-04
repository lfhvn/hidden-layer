import { useState, useCallback, useRef, useEffect } from 'react'

export interface StreamMessage {
  type: 'status' | 'agent' | 'judge' | 'synthesis' | 'complete' | 'error'
  content: string
  agent_id?: string
  role?: string
  metadata?: Record<string, any>
}

interface UseDebateStreamResult {
  messages: StreamMessage[]
  isStreaming: boolean
  error: string | null
  startStream: (question: string, strategy: string, nAgents: number) => void
  stopStream: () => void
  clearMessages: () => void
}

export function useDebateStream(): UseDebateStreamResult {
  const [messages, setMessages] = useState<StreamMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  const stopStream = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsStreaming(false)
  }, [])

  const startStream = useCallback((question: string, strategy: string, nAgents: number) => {
    // Clear previous state
    setMessages([])
    setError(null)
    setIsStreaming(true)

    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close()
    }

    // Determine WebSocket URL
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'
    const ws = new WebSocket(`${wsUrl}/ws/debate`)

    ws.onopen = () => {
      console.log('WebSocket connected')
      // Send request
      ws.send(JSON.stringify({
        question,
        strategy,
        n_agents: nAgents
      }))
    }

    ws.onmessage = (event) => {
      try {
        const message: StreamMessage = JSON.parse(event.data)
        console.log('Received message:', message)

        setMessages(prev => [...prev, message])

        // If complete or error, stop streaming
        if (message.type === 'complete' || message.type === 'error') {
          setIsStreaming(false)
        }
      } catch (err) {
        console.error('Failed to parse message:', err)
      }
    }

    ws.onerror = (event) => {
      console.error('WebSocket error:', event)
      setError('Connection error. Please try again.')
      setIsStreaming(false)
    }

    ws.onclose = () => {
      console.log('WebSocket closed')
      setIsStreaming(false)
    }

    wsRef.current = ws
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
    setError(null)
  }, [])

  return {
    messages,
    isStreaming,
    error,
    startStream,
    stopStream,
    clearMessages
  }
}
