'use client'

import type { Feature } from '@/types'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from './ui/card'
import { Badge } from './ui/badge'
import { formatNumber, getSparsityColor } from '@/lib/utils'

interface FeatureCardProps {
  feature: Feature
  onClick?: () => void
}

export function FeatureCard({ feature, onClick }: FeatureCardProps) {
  return (
    <Card
      className="cursor-pointer hover:shadow-lg transition-shadow"
      onClick={onClick}
    >
      <CardHeader>
        <CardTitle className="text-lg">Feature #{feature.feature_index}</CardTitle>
        <CardDescription>
          Experiment {feature.experiment_id}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className="text-muted-foreground">Mean:</span>
            <span className="ml-2 font-mono">{formatNumber(feature.activation_mean)}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Max:</span>
            <span className="ml-2 font-mono">{formatNumber(feature.activation_max)}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Std:</span>
            <span className="ml-2 font-mono">{formatNumber(feature.activation_std)}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Sparsity:</span>
            <span className={`ml-2 font-mono ${getSparsityColor(feature.sparsity)}`}>
              {formatNumber(feature.sparsity)}
            </span>
          </div>
        </div>

        {feature.top_tokens.length > 0 && (
          <div>
            <div className="text-sm text-muted-foreground mb-2">Top Tokens:</div>
            <div className="flex flex-wrap gap-1">
              {feature.top_tokens.slice(0, 5).map((token, idx) => (
                <Badge key={idx} variant="secondary">
                  {token}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
