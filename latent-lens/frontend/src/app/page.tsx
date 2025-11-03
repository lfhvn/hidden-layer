'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api'
import type { Experiment } from '@/types'
import { formatDate, getStatusColor } from '@/lib/utils'

export default function HomePage() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadExperiments()
  }, [])

  const loadExperiments = async () => {
    try {
      setLoading(true)
      const data = await api.listExperiments(10)
      setExperiments(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load experiments')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold mb-2">Latent Lens</h1>
        <p className="text-lg text-muted-foreground">
          Interactive LLM Interpretability with Sparse Autoencoders
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Layer Explorer</CardTitle>
            <CardDescription>
              Choose layers and train SAEs
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/layer-explorer">
              <Button className="w-full">Explore Layers</Button>
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Feature Gallery</CardTitle>
            <CardDescription>
              Browse discovered features
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/feature-gallery">
              <Button className="w-full">View Features</Button>
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Activation Lens</CardTitle>
            <CardDescription>
              Analyze text activations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/activation-lens">
              <Button className="w-full">Analyze Text</Button>
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Labeling</CardTitle>
            <CardDescription>
              Label and group features
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/labeling">
              <Button className="w-full">Label Features</Button>
            </Link>
          </CardContent>
        </Card>
      </div>

      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">Recent Experiments</h2>
          <Button onClick={loadExperiments} variant="outline" size="sm">
            Refresh
          </Button>
        </div>

        {loading && <p className="text-muted-foreground">Loading experiments...</p>}
        {error && <p className="text-destructive">Error: {error}</p>}

        {!loading && !error && experiments.length === 0 && (
          <p className="text-muted-foreground">
            No experiments yet. Start by exploring layers!
          </p>
        )}

        <div className="grid gap-4">
          {experiments.map((exp) => (
            <Card key={exp.id}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>{exp.name}</CardTitle>
                  <Badge className={getStatusColor(exp.status)}>
                    {exp.status}
                  </Badge>
                </div>
                <CardDescription>
                  {exp.model_name} - {exp.layer_name}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Hidden Dim:</span>
                    <span className="ml-2 font-mono">{exp.hidden_dim}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Samples:</span>
                    <span className="ml-2 font-mono">{exp.num_samples}</span>
                  </div>
                  {exp.train_loss && (
                    <div>
                      <span className="text-muted-foreground">Loss:</span>
                      <span className="ml-2 font-mono">{exp.train_loss.toFixed(4)}</span>
                    </div>
                  )}
                  <div>
                    <span className="text-muted-foreground">Created:</span>
                    <span className="ml-2 text-xs">{formatDate(exp.created_at)}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
