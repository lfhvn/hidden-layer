'use client'

import { useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { FeatureCard } from '@/components/FeatureCard'
import { LabelEditor } from '@/components/LabelEditor'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api'
import type { Feature, FeatureLabel, LabelCreate } from '@/types'

export default function LabelingPage() {
  const [features, setFeatures] = useState<Feature[]>([])
  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null)
  const [labels, setLabels] = useState<FeatureLabel[]>([])
  const [experimentId, setExperimentId] = useState('')
  const [loading, setLoading] = useState(false)

  const loadFeatures = async () => {
    if (!experimentId) return

    setLoading(true)
    try {
      const data = await api.listFeatures({
        experiment_id: parseInt(experimentId),
        limit: 20,
      })
      setFeatures(data)
    } catch (err) {
      console.error('Failed to load features:', err)
    } finally {
      setLoading(false)
    }
  }

  const loadLabels = async (featureId: number) => {
    try {
      const data = await api.getLabels(featureId)
      setLabels(data)
    } catch (err) {
      console.error('Failed to load labels:', err)
    }
  }

  const handleFeatureSelect = async (feature: Feature) => {
    setSelectedFeature(feature)
    await loadLabels(feature.id)
  }

  const handleAddLabel = async (labelData: LabelCreate) => {
    if (!selectedFeature) return

    try {
      await api.addLabel(selectedFeature.id, labelData)
      await loadLabels(selectedFeature.id)
    } catch (err) {
      console.error('Failed to add label:', err)
      throw err
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold mb-2">Feature Labeling</h1>
        <p className="text-muted-foreground">
          Add labels and annotations to discovered features
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Select Experiment</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <div className="flex-1">
              <Input
                type="number"
                value={experimentId}
                onChange={(e) => setExperimentId(e.target.value)}
                placeholder="Enter Experiment ID"
              />
            </div>
            <Button onClick={loadFeatures} disabled={loading || !experimentId}>
              {loading ? 'Loading...' : 'Load Features'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {features.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h2 className="text-xl font-semibold mb-4">Features</h2>
            <div className="space-y-3 max-h-[600px] overflow-y-auto">
              {features.map((feature) => (
                <div
                  key={feature.id}
                  className={`cursor-pointer transition-all ${
                    selectedFeature?.id === feature.id
                      ? 'ring-2 ring-primary'
                      : ''
                  }`}
                  onClick={() => handleFeatureSelect(feature)}
                >
                  <FeatureCard feature={feature} />
                </div>
              ))}
            </div>
          </div>

          <div>
            <h2 className="text-xl font-semibold mb-4">
              {selectedFeature
                ? `Labels for Feature #${selectedFeature.feature_index}`
                : 'Select a Feature'}
            </h2>

            {selectedFeature && (
              <div className="space-y-4">
                {labels.length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Existing Labels</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      {labels.map((label) => (
                        <div key={label.id} className="p-3 border rounded-md">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-semibold">{label.label}</span>
                            <span className="text-sm text-muted-foreground">
                              Confidence: {label.confidence.toFixed(2)}
                            </span>
                          </div>
                          {label.description && (
                            <p className="text-sm text-muted-foreground mb-2">
                              {label.description}
                            </p>
                          )}
                          {label.tags.length > 0 && (
                            <div className="flex flex-wrap gap-1">
                              {label.tags.map((tag, idx) => (
                                <Badge key={idx} variant="outline">
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                )}

                <LabelEditor
                  featureId={selectedFeature.id}
                  onSubmit={handleAddLabel}
                />
              </div>
            )}

            {!selectedFeature && (
              <p className="text-muted-foreground">
                Click on a feature to view and add labels
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
