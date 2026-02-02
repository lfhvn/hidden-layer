'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { workflowsApi, Workflow } from '@/lib/api'
import { PlayCircle, Eye, Trash2, Plus } from 'lucide-react'

// Default org ID (in real app, this comes from auth)
const DEFAULT_ORG_ID = '00000000-0000-0000-0000-000000000000'

export default function WorkflowsPage() {
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadWorkflows()
  }, [])

  async function loadWorkflows() {
    try {
      setLoading(true)
      const data = await workflowsApi.list(DEFAULT_ORG_ID)
      setWorkflows(data)
      setError(null)
    } catch (err: any) {
      setError(err.message || 'Failed to load workflows')
    } finally {
      setLoading(false)
    }
  }

  async function handleDelete(workflowId: string) {
    if (!confirm('Are you sure you want to delete this workflow?')) {
      return
    }

    try {
      await workflowsApi.delete(workflowId)
      await loadWorkflows()
    } catch (err: any) {
      alert(`Failed to delete: ${err.message}`)
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading workflows...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="text-red-800 font-medium">Error loading workflows</div>
        <div className="text-red-600 text-sm mt-1">{error}</div>
        <button
          onClick={loadWorkflows}
          className="mt-3 text-sm text-red-600 hover:text-red-800 font-medium"
        >
          Try again
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Workflows</h1>
          <p className="text-gray-600 mt-1">
            Create and manage multi-agent workflows
          </p>
        </div>
        <Link
          href="/workflows/new"
          className="bg-indigo-600 text-white px-4 py-2 rounded-md font-medium hover:bg-indigo-700 transition flex items-center gap-2"
        >
          <Plus size={20} />
          New Workflow
        </Link>
      </div>

      {/* Workflows list */}
      {workflows.length === 0 ? (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
          <div className="text-gray-400 text-5xl mb-4">🔄</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No workflows yet
          </h3>
          <p className="text-gray-600 mb-4">
            Get started by creating your first workflow
          </p>
          <Link
            href="/workflows/new"
            className="inline-block bg-indigo-600 text-white px-6 py-2 rounded-md font-medium hover:bg-indigo-700 transition"
          >
            Create Workflow
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {workflows.map((workflow) => (
            <div
              key={workflow.id}
              className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition"
            >
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900">
                    {workflow.name}
                  </h3>
                  {workflow.description && (
                    <p className="text-sm text-gray-600 mt-1">
                      {workflow.description}
                    </p>
                  )}
                  <div className="flex gap-4 mt-3 text-sm text-gray-500">
                    <div>
                      {workflow.graph.nodes.length} node
                      {workflow.graph.nodes.length !== 1 ? 's' : ''}
                    </div>
                    <div>
                      {workflow.graph.edges.length} edge
                      {workflow.graph.edges.length !== 1 ? 's' : ''}
                    </div>
                    <div>
                      Created{' '}
                      {new Date(workflow.created_at).toLocaleDateString()}
                    </div>
                  </div>

                  {/* Strategy nodes preview */}
                  <div className="flex gap-2 mt-3 flex-wrap">
                    {workflow.graph.nodes
                      .filter((n) => n.type === 'strategy')
                      .map((node) => (
                        <span
                          key={node.id}
                          className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800"
                        >
                          {node.strategy_id}
                        </span>
                      ))}
                  </div>
                </div>

                <div className="flex gap-2 ml-4">
                  <Link
                    href={`/workflows/${workflow.id}`}
                    className="p-2 text-gray-600 hover:text-indigo-600 hover:bg-indigo-50 rounded transition"
                    title="View details"
                  >
                    <Eye size={20} />
                  </Link>
                  <Link
                    href={`/workflows/${workflow.id}/run`}
                    className="p-2 text-gray-600 hover:text-green-600 hover:bg-green-50 rounded transition"
                    title="Execute workflow"
                  >
                    <PlayCircle size={20} />
                  </Link>
                  <button
                    onClick={() => handleDelete(workflow.id)}
                    className="p-2 text-gray-600 hover:text-red-600 hover:bg-red-50 rounded transition"
                    title="Delete workflow"
                  >
                    <Trash2 size={20} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
