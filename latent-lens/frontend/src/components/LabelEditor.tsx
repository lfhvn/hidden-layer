'use client'

import { useState } from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import type { LabelCreate } from '@/types'

interface LabelEditorProps {
  featureId: number
  onSubmit: (label: LabelCreate) => Promise<void>
  onCancel?: () => void
}

export function LabelEditor({ featureId, onSubmit, onCancel }: LabelEditorProps) {
  const [label, setLabel] = useState('')
  const [description, setDescription] = useState('')
  const [tags, setTags] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async () => {
    if (!label.trim()) return

    setIsSubmitting(true)
    try {
      await onSubmit({
        label: label.trim(),
        description: description.trim() || undefined,
        tags: tags.split(',').map(t => t.trim()).filter(Boolean),
      })

      // Reset form
      setLabel('')
      setDescription('')
      setTags('')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="space-y-4 border rounded-lg p-4">
      <h3 className="font-semibold">Add Label to Feature #{featureId}</h3>

      <div>
        <label className="text-sm font-medium mb-2 block">Label *</label>
        <Input
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="e.g., 'sentiment', 'negation'"
          disabled={isSubmitting}
        />
      </div>

      <div>
        <label className="text-sm font-medium mb-2 block">Description</label>
        <Input
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Detailed description of what this feature represents"
          disabled={isSubmitting}
        />
      </div>

      <div>
        <label className="text-sm font-medium mb-2 block">Tags (comma-separated)</label>
        <Input
          value={tags}
          onChange={(e) => setTags(e.target.value)}
          placeholder="e.g., syntax, semantic, attention"
          disabled={isSubmitting}
        />
      </div>

      <div className="flex gap-2">
        <Button
          onClick={handleSubmit}
          disabled={!label.trim() || isSubmitting}
          className="flex-1"
        >
          {isSubmitting ? 'Submitting...' : 'Add Label'}
        </Button>
        {onCancel && (
          <Button
            onClick={onCancel}
            variant="outline"
            disabled={isSubmitting}
          >
            Cancel
          </Button>
        )}
      </div>
    </div>
  )
}
