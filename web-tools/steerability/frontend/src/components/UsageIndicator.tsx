'use client'

import { useEffect, useState } from 'react'

interface Usage {
  limit: number
  used: number
  remaining: number
  reset: number
  window: number
}

export function UsageIndicator() {
  const [usage, setUsage] = useState<Usage | null>(null)
  const [timeUntilReset, setTimeUntilReset] = useState<string>('')

  useEffect(() => {
    fetchUsage()
    const interval = setInterval(fetchUsage, 10000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (!usage) return

    const interval = setInterval(() => {
      const now = Math.floor(Date.now() / 1000)
      const seconds = usage.reset - now

      if (seconds <= 0) {
        setTimeUntilReset('Resetting...')
        fetchUsage()
      } else {
        const minutes = Math.floor(seconds / 60)
        const secs = seconds % 60
        setTimeUntilReset(`${minutes}m ${secs}s`)
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [usage])

  const fetchUsage = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/usage')
      const data = await response.json()
      setUsage(data)
    } catch (error) {
      console.error('Failed to fetch usage:', error)
    }
  }

  if (!usage) return null

  const percentage = (usage.used / usage.limit) * 100

  return (
    <div className="bg-white rounded-lg shadow p-4 mb-6">
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm font-medium text-gray-700">
          Rate Limit Status
        </div>
        <div className="text-xs text-gray-500">
          Resets in {timeUntilReset}
        </div>
      </div>

      <div className="w-full bg-gray-200 rounded-full h-2.5 mb-2">
        <div
          className={`h-2.5 rounded-full transition-all ${
            percentage >= 100
              ? 'bg-red-600'
              : percentage >= 66
              ? 'bg-yellow-600'
              : 'bg-green-600'
          }`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>

      <div className="flex justify-between text-sm">
        <span className="text-gray-600">
          {usage.used} / {usage.limit} requests used
        </span>
        <span className={`font-medium ${
          usage.remaining > 0 ? 'text-green-600' : 'text-red-600'
        }`}>
          {usage.remaining} remaining
        </span>
      </div>

      {usage.remaining === 0 && (
        <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
          ⚠️ Rate limit reached. Reset in {timeUntilReset}.
        </div>
      )}
    </div>
  )
}
