'use client'

import { useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { FeatureCard } from '@/components/FeatureCard'
import { api } from '@/lib/api'
import type { Feature } from '@/types'

export default function FeatureGalleryPage() {
  const [features, setFeatures] = useState<Feature[]>([])
  const [loading, setLoading] = useState(false)
  const [experimentId, setExperimentId] = useState('')
  const [minSparsity, setMinSparsity] = useState('')
  const [maxSparsity, setMaxSparsity] = useState('')

  const loadFeatures = async () => {
    setLoading(true)
    try {
      const data = await api.listFeatures({
        experiment_id: experimentId ? parseInt(experimentId) : undefined,
        min_sparsity: minSparsity ? parseFloat(minSparsity) : undefined,
        max_sparsity: maxSparsity ? parseFloat(maxSparsity) : undefined,
        limit: 50,
      })
      setFeatures(data)
    } catch (err) {
      console.error('Failed to load features:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadFeatures()
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Feature Gallery</h1>
        <p className="text-muted-foreground">
          Browse and explore discovered features from trained SAEs
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Filters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Experiment ID</label>
              <Input
                type="number"
                value={experimentId}
                onChange={(e) => setExperimentId(e.target.value)}
                placeholder="All experiments"
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Min Sparsity</label>
              <Input
                type="number"
                step="0.01"
                value={minSparsity}
                onChange={(e) => setMinSparsity(e.target.value)}
                placeholder="0.0"
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Max Sparsity</label>
              <Input
                type="number"
                step="0.01"
                value={maxSparsity}
                onChange={(e) => setMaxSparsity(e.target.value)}
                placeholder="1.0"
              />
            </div>

            <div className="flex items-end">
              <Button onClick={loadFeatures} disabled={loading} className="w-full">
                {loading ? 'Loading...' : 'Apply Filters'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">
            {features.length} Feature{features.length !== 1 ? 's' : ''}
          </h2>
        </div>

        {loading && (
          <p className="text-muted-foreground">Loading features...</p>
        )}

        {!loading && features.length === 0 && (
          <p className="text-muted-foreground">
            No features found. Try adjusting your filters or create an experiment first.
          </p>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((feature) => (
            <FeatureCard
              key={feature.id}
              feature={feature}
              onClick={() => {
                // Could navigate to detail view
                console.log('Feature clicked:', feature.id)
              }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
