'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { Clock, CheckCircle, XCircle, Loader, Eye } from 'lucide-react'

// TODO: Add runs list API endpoint to backend
// For now, this is a placeholder that shows the structure

interface RunListItem {
  id: string
  workflow_id: string
  workflow_name: string
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'canceled'
  created_at: string
  finished_at?: string
}

export default function RunsPage() {
  const [runs, setRuns] = useState<RunListItem[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // TODO: Implement when backend adds /api/runs endpoint
    setLoading(false)
  }, [])

  const statusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="text-gray-500" size={20} />
      case 'running':
        return <Loader className="text-blue-500 animate-spin" size={20} />
      case 'succeeded':
        return <CheckCircle className="text-green-500" size={20} />
      case 'failed':
        return <XCircle className="text-red-500" size={20} />
      default:
        return <Clock className="text-gray-500" size={20} />
    }
  }

  const statusColor = (status: string) => {
    switch (status) {
      case 'pending':
        return 'bg-gray-100 text-gray-800'
      case 'running':
        return 'bg-blue-100 text-blue-800'
      case 'succeeded':
        return 'bg-green-100 text-green-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading runs...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Workflow Runs</h1>
        <p className="text-gray-600 mt-1">
          View execution history and results
        </p>
      </div>

      {/* Coming soon notice */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
        <div className="flex items-start gap-3">
          <div className="text-2xl">🚧</div>
          <div>
            <h3 className="font-medium text-yellow-900">Coming Soon</h3>
            <p className="text-sm text-yellow-800 mt-1">
              The runs list page is under development. For now, you can:
            </p>
            <ul className="text-sm text-yellow-800 mt-2 space-y-1 list-disc list-inside">
              <li>Execute workflows from the workflow detail page</li>
              <li>View individual run details if you have the run ID</li>
              <li>Check the API docs at <a href="http://localhost:8000/docs" target="_blank" className="underline">localhost:8000/docs</a></li>
            </ul>
          </div>
        </div>
      </div>

      {/* Empty state */}
      {runs.length === 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
          <div className="text-gray-400 text-5xl mb-4">🏃</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No runs yet
          </h3>
          <p className="text-gray-600 mb-4">
            Execute a workflow to see it here
          </p>
          <Link
            href="/workflows"
            className="inline-block bg-indigo-600 text-white px-6 py-2 rounded-md font-medium hover:bg-indigo-700 transition"
          >
            View Workflows
          </Link>
        </div>
      )}

      {/* Runs list (when implemented) */}
      {runs.length > 0 && (
        <div className="grid grid-cols-1 gap-4">
          {runs.map((run) => (
            <div
              key={run.id}
              className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition"
            >
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    {statusIcon(run.status)}
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900">
                        {run.workflow_name}
                      </h3>
                      <div className="text-sm text-gray-500 mt-1">
                        Run ID: {run.id.slice(0, 8)}...
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-4 mt-3 text-sm text-gray-500">
                    <div>
                      Started: {new Date(run.created_at).toLocaleString()}
                    </div>
                    {run.finished_at && (
                      <div>
                        Finished: {new Date(run.finished_at).toLocaleString()}
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-3 ml-4">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${statusColor(run.status)}`}>
                    {run.status}
                  </span>
                  <Link
                    href={`/runs/${run.id}`}
                    className="p-2 text-gray-600 hover:text-indigo-600 hover:bg-indigo-50 rounded transition"
                    title="View details"
                  >
                    <Eye size={20} />
                  </Link>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
