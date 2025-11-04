'use client'

import { useEffect, useState } from 'react'

interface Strategy {
  name: string
  display_name: string
  description: string
  best_for: string
  typical_agents: number
}

interface StrategySelectorProps {
  onSelect: (strategy: Strategy) => void
  selected: Strategy | null
}

export function StrategySelector({ onSelect, selected }: StrategySelectorProps) {
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('http://localhost:8000/api/strategies')
      .then(res => res.json())
      .then(data => {
        setStrategies(data.strategies)
        // Auto-select first strategy
        if (data.strategies.length > 0 && !selected) {
          onSelect(data.strategies[0])
        }
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to load strategies:', err)
        setLoading(false)
      })
  }, [])

  if (loading) {
    return <div className="animate-pulse bg-gray-200 h-12 rounded"></div>
  }

  return (
    <div className="space-y-2">
      {strategies.map((strategy) => (
        <button
          key={strategy.name}
          onClick={() => onSelect(strategy)}
          className={`w-full text-left p-3 rounded-lg border-2 transition-all ${
            selected?.name === strategy.name
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-200 hover:border-gray-300'
          }`}
        >
          <div className="font-semibold">{strategy.display_name}</div>
          <div className="text-xs text-gray-600 mt-1">
            {strategy.description}
          </div>
        </button>
      ))}
    </div>
  )
}
