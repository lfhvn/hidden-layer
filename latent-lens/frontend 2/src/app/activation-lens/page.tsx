'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ActivationHeatmap } from '@/components/ActivationHeatmap'
import { api } from '@/lib/api'
import type { AnalyzeResponse } from '@/types'

export default function ActivationLensPage() {
  const [text, setText] = useState('The quick brown fox jumps over the lazy dog.')
  const [experimentId, setExperimentId] = useState('1')
  const [topK, setTopK] = useState(10)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = async () => {
    if (!text.trim() || !experimentId) {
      setError('Please provide text and experiment ID')
      return
    }

    setIsAnalyzing(true)
    setError(null)

    try {
      const data = await api.analyzeText(text, parseInt(experimentId), topK)
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze text')
      setResult(null)
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Activation Lens</h1>
        <p className="text-muted-foreground">
          Analyze which features activate for specific text inputs
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Input Text</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-sm font-medium mb-2 block">Text to Analyze</label>
            <textarea
              className="w-full min-h-[100px] p-3 border rounded-md resize-y"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text to analyze..."
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Experiment ID</label>
              <Input
                type="number"
                value={experimentId}
                onChange={(e) => setExperimentId(e.target.value)}
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Top-K Features</label>
              <Input
                type="number"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                min={1}
                max={50}
              />
            </div>

            <div className="flex items-end">
              <Button
                onClick={handleAnalyze}
                disabled={isAnalyzing || !text.trim()}
                className="w-full"
              >
                {isAnalyzing ? 'Analyzing...' : 'Analyze'}
              </Button>
            </div>
          </div>

          {error && (
            <div className="p-3 bg-red-50 text-red-800 border border-red-200 rounded-md text-sm">
              {error}
            </div>
          )}
        </CardContent>
      </Card>

      {result && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Token Activations</CardTitle>
            </CardHeader>
            <CardContent>
              <ActivationHeatmap tokens={result.tokens} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Top Features</CardTitle>
            </CardHeader>
            <CardContent>
              {result.top_features.length > 0 ? (
                <div className="space-y-2">
                  {result.top_features.map(([featureId, score], idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between p-2 border rounded"
                    >
                      <span className="font-mono">Feature #{featureId}</span>
                      <span className="text-sm text-muted-foreground">
                        Score: {score.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-muted-foreground">No active features detected</p>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
