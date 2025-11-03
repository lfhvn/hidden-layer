'use client'

import { useMemo } from 'react'
import type { TokenActivation } from '@/types'

interface ActivationHeatmapProps {
  tokens: TokenActivation[]
  selectedFeature?: number
}

export function ActivationHeatmap({ tokens, selectedFeature }: ActivationHeatmapProps) {
  const getIntensity = (activation: number): string => {
    const normalized = Math.min(activation, 1.0)
    const opacity = Math.floor(normalized * 255).toString(16).padStart(2, '0')
    return `#3b82f6${opacity}` // blue with varying opacity
  }

  const tokenData = useMemo(() => {
    return tokens.map(t => {
      if (selectedFeature !== undefined) {
        const feature = t.features.find(f => f.feature_id === selectedFeature)
        return {
          token: t.token,
          activation: feature?.activation_value || 0
        }
      }
      // Use max activation across all features
      const maxActivation = Math.max(...t.features.map(f => f.activation_value), 0)
      return {
        token: t.token,
        activation: maxActivation
      }
    })
  }, [tokens, selectedFeature])

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1">
        {tokenData.map((data, idx) => (
          <div
            key={idx}
            className="px-2 py-1 rounded text-sm font-mono border"
            style={{
              backgroundColor: getIntensity(data.activation),
              color: data.activation > 0.5 ? 'white' : 'black'
            }}
            title={`Activation: ${data.activation.toFixed(3)}`}
          >
            {data.token}
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <span>Low</span>
        <div className="h-4 w-32 rounded" style={{
          background: 'linear-gradient(to right, #3b82f610, #3b82f6ff)'
        }} />
        <span>High</span>
      </div>
    </div>
  )
}
