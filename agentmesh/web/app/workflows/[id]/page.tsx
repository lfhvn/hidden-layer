'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { workflowsApi, Workflow } from '@/lib/api'
import { ArrowLeft, PlayCircle, Edit, Trash2, Eye } from 'lucide-react'

export default function WorkflowDetailPage() {
  const params = useParams()
  const router = useRouter()
  const workflowId = params.id as string

  const [workflow, setWorkflow] = useState<Workflow | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadWorkflow()
  }, [workflowId])

  async function loadWorkflow() {
    try {
      setLoading(true)
      const data = await workflowsApi.get(workflowId)
      setWorkflow(data)
      setError(null)
    } catch (err: any) {
      setError(err.message || 'Failed to load workflow')
    } finally {
      setLoading(false)
    }
  }

  async function handleDelete() {
    if (!confirm('Are you sure you want to delete this workflow?')) {
      return
    }

    try {
      await workflowsApi.delete(workflowId)
      router.push('/workflows')
    } catch (err: any) {
      alert(`Failed to delete: ${err.message}`)
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading workflow...</div>
      </div>
    )
  }

  if (error || !workflow) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="text-red-800 font-medium">Error loading workflow</div>
        <div className="text-red-600 text-sm mt-1">{error || 'Workflow not found'}</div>
      </div>
    )
  }

  const strategyNodes = workflow.graph.nodes.filter((n) => n.type === 'strategy')

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link
          href="/workflows"
          className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition"
        >
          <ArrowLeft size={20} />
        </Link>
        <div className="flex-1">
          <h1 className="text-3xl font-bold text-gray-900">{workflow.name}</h1>
          {workflow.description && (
            <p className="text-gray-600 mt-1">{workflow.description}</p>
          )}
        </div>
        <div className="flex gap-2">
          <Link
            href={`/workflows/${workflowId}/run`}
            className="bg-indigo-600 text-white px-6 py-2 rounded-md font-medium hover:bg-indigo-700 transition flex items-center gap-2"
          >
            <PlayCircle size={20} />
            Execute
          </Link>
          <button
            onClick={handleDelete}
            className="p-2 text-gray-600 hover:text-red-600 hover:bg-red-50 rounded transition"
            title="Delete workflow"
          >
            <Trash2 size={20} />
          </button>
        </div>
      </div>

      {/* Metadata */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="text-sm text-gray-500">Nodes</div>
          <div className="text-2xl font-bold text-gray-900">
            {workflow.graph.nodes.length}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {strategyNodes.length} strategies
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="text-sm text-gray-500">Connections</div>
          <div className="text-2xl font-bold text-gray-900">
            {workflow.graph.edges.length}
          </div>
          <div className="text-xs text-gray-500 mt-1">edges</div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="text-sm text-gray-500">Created</div>
          <div className="text-lg font-semibold text-gray-900">
            {new Date(workflow.created_at).toLocaleDateString()}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {new Date(workflow.created_at).toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Strategy Nodes */}
      {strategyNodes.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Strategies Used
          </h2>
          <div className="space-y-3">
            {strategyNodes.map((node) => (
              <div
                key={node.id}
                className="flex items-start gap-4 p-4 border border-gray-200 rounded-lg bg-gray-50"
              >
                <div className="flex-1">
                  <div className="font-medium text-gray-900">{node.label}</div>
                  <div className="text-sm text-gray-600 mt-1">
                    Strategy: <span className="font-mono text-indigo-600">{node.strategy_id}</span>
                  </div>
                  {Object.keys(node.config || {}).length > 0 && (
                    <details className="mt-2">
                      <summary className="text-sm text-gray-700 cursor-pointer">
                        Configuration
                      </summary>
                      <pre className="text-xs bg-white p-2 rounded border border-gray-200 mt-1 overflow-auto">
                        {JSON.stringify(node.config, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Graph Structure */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Workflow Graph
        </h2>

        {/* Nodes */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Nodes</h3>
          <div className="grid grid-cols-2 gap-3">
            {workflow.graph.nodes.map((node) => (
              <div
                key={node.id}
                className="flex items-center gap-3 p-3 border border-gray-200 rounded-lg bg-gray-50"
              >
                <div className="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600 font-semibold text-sm">
                  {node.type === 'start' && '▶'}
                  {node.type === 'end' && '⏹'}
                  {node.type === 'strategy' && '⚡'}
                </div>
                <div className="flex-1">
                  <div className="font-medium text-gray-900">{node.label}</div>
                  <div className="text-xs text-gray-500">
                    {node.type}
                    {node.strategy_id && ` • ${node.strategy_id}`}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Edges */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-3">Connections</h3>
          <div className="space-y-2">
            {workflow.graph.edges.map((edge) => {
              const fromNode = workflow.graph.nodes.find(
                (n) => n.id === edge.from_node_id
              )
              const toNode = workflow.graph.nodes.find(
                (n) => n.id === edge.to_node_id
              )
              return (
                <div
                  key={edge.id}
                  className="flex items-center gap-3 p-3 border border-gray-200 rounded-lg bg-gray-50"
                >
                  <div className="flex-1 text-sm">
                    <span className="font-medium text-gray-900">
                      {fromNode?.label || edge.from_node_id}
                    </span>
                    <span className="text-gray-500 mx-3">→</span>
                    <span className="font-medium text-gray-900">
                      {toNode?.label || edge.to_node_id}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Raw JSON (for debugging) */}
      <details className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <summary className="text-lg font-semibold text-gray-900 cursor-pointer">
          Raw Workflow Data (JSON)
        </summary>
        <pre className="text-xs bg-gray-50 p-4 rounded border border-gray-200 mt-4 overflow-auto max-h-96">
          {JSON.stringify(workflow, null, 2)}
        </pre>
      </details>
    </div>
  )
}
