'use client'

import { useState } from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'

interface LayerSelectorProps {
  modelName: string
  onSelect: (layerName: string, layerIndex: number) => void
}

export function LayerSelector({ modelName, onSelect }: LayerSelectorProps) {
  const [layerIndex, setLayerIndex] = useState(6)

  const handleSelect = () => {
    const layerName = `transformer.h.${layerIndex}.mlp`
    onSelect(layerName, layerIndex)
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="text-sm font-medium mb-2 block">
          Model: <span className="font-mono">{modelName}</span>
        </label>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex-1">
          <label className="text-sm font-medium mb-2 block">Layer Index</label>
          <Input
            type="number"
            min={0}
            max={11}
            value={layerIndex}
            onChange={(e) => setLayerIndex(parseInt(e.target.value))}
            className="w-full"
          />
        </div>

        <div className="flex-1">
          <label className="text-sm font-medium mb-2 block">Component</label>
          <div className="text-sm font-mono bg-muted px-3 py-2 rounded-md">
            transformer.h.{layerIndex}.mlp
          </div>
        </div>
      </div>

      <Button onClick={handleSelect} className="w-full">
        Select Layer
      </Button>
    </div>
  )
}
