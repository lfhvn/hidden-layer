'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { LayerSelector } from '@/components/LayerSelector'
import { api } from '@/lib/api'
import type { ExperimentCreate } from '@/types'

export default function LayerExplorerPage() {
  const [modelName, setModelName] = useState('gpt2')
  const [experimentName, setExperimentName] = useState('')
  const [selectedLayer, setSelectedLayer] = useState<{ name: string; index: number } | null>(null)
  const [hiddenDim, setHiddenDim] = useState(4096)
  const [sparsityCoef, setSparsityCoef] = useState(0.01)
  const [isCreating, setIsCreating] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const handleLayerSelect = (layerName: string, layerIndex: number) => {
    setSelectedLayer({ name: layerName, index: layerIndex })
    setMessage(null)
  }

  const handleCreateExperiment = async () => {
    if (!selectedLayer || !experimentName.trim()) {
      setMessage({ type: 'error', text: 'Please select a layer and provide an experiment name' })
      return
    }

    setIsCreating(true)
    setMessage(null)

    try {
      const data: ExperimentCreate = {
        name: experimentName.trim(),
        model_name: modelName,
        layer_name: selectedLayer.name,
        layer_index: selectedLayer.index,
        input_dim: 768, // GPT-2 hidden size
        hidden_dim: hiddenDim,
        sparsity_coef: sparsityCoef,
        learning_rate: 1e-3,
        num_epochs: 10,
      }

      const experiment = await api.createExperiment(data)

      setMessage({
        type: 'success',
        text: `Experiment "${experiment.name}" created successfully! ID: ${experiment.id}`,
      })

      // Reset form
      setExperimentName('')
      setSelectedLayer(null)
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Failed to create experiment',
      })
    } finally {
      setIsCreating(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Layer Explorer</h1>
        <p className="text-muted-foreground">
          Select a model layer to train a Sparse Autoencoder
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Model Configuration</CardTitle>
            <CardDescription>
              Configure the model and layer to analyze
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Model Name</label>
              <Input
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="e.g., gpt2, gpt2-medium"
              />
            </div>

            <LayerSelector
              modelName={modelName}
              onSelect={handleLayerSelect}
            />

            {selectedLayer && (
              <div className="p-3 bg-muted rounded-md">
                <p className="text-sm">
                  <span className="font-medium">Selected:</span>{' '}
                  <span className="font-mono">{selectedLayer.name}</span>
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>SAE Configuration</CardTitle>
            <CardDescription>
              Configure the Sparse Autoencoder
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Experiment Name *</label>
              <Input
                value={experimentName}
                onChange={(e) => setExperimentName(e.target.value)}
                placeholder="e.g., gpt2_layer6_mlp"
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Hidden Dimension</label>
              <Input
                type="number"
                value={hiddenDim}
                onChange={(e) => setHiddenDim(parseInt(e.target.value))}
                min={256}
                max={16384}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Number of features to learn (typically 4-8x input dim)
              </p>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Sparsity Coefficient</label>
              <Input
                type="number"
                step="0.001"
                value={sparsityCoef}
                onChange={(e) => setSparsityCoef(parseFloat(e.target.value))}
                min={0.001}
                max={0.1}
              />
              <p className="text-xs text-muted-foreground mt-1">
                L1 penalty strength (higher = sparser)
              </p>
            </div>

            <Button
              onClick={handleCreateExperiment}
              disabled={!selectedLayer || !experimentName.trim() || isCreating}
              className="w-full"
            >
              {isCreating ? 'Creating...' : 'Create Experiment'}
            </Button>

            {message && (
              <div
                className={`p-3 rounded-md text-sm ${
                  message.type === 'success'
                    ? 'bg-green-50 text-green-800 border border-green-200'
                    : 'bg-red-50 text-red-800 border border-red-200'
                }`}
              >
                {message.text}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>About Layer Explorer</CardTitle>
        </CardHeader>
        <CardContent className="prose prose-sm max-w-none">
          <p>
            The Layer Explorer allows you to select specific layers from a language model
            and train Sparse Autoencoders (SAEs) to discover interpretable features.
          </p>
          <ul>
            <li>Choose a model and specific layer/component to analyze</li>
            <li>Configure SAE architecture (hidden dimension, sparsity)</li>
            <li>Create experiments that can be trained on text data</li>
            <li>Monitor training progress and view discovered features</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
