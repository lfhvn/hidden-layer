'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { workflowsApi, WorkflowNode, WorkflowEdge } from '@/lib/api'
import { ArrowLeft, Plus, Trash2, Save } from 'lucide-react'
import Link from 'next/link'

// Default org ID
const DEFAULT_ORG_ID = '00000000-0000-0000-0000-000000000000'

// Available strategies from Hidden Layer
const STRATEGIES = [
  { id: 'debate', name: 'Debate', emoji: '💬', config: { n_debaters: 3, n_rounds: 2 } },
  { id: 'crit', name: 'CRIT', emoji: '🎨', config: {} },
  { id: 'consensus', name: 'Consensus', emoji: '🤝', config: { n_agents: 3 } },
  { id: 'manager_worker', name: 'Manager-Worker', emoji: '👔', config: { n_workers: 3 } },
  { id: 'self_consistency', name: 'Self-Consistency', emoji: '🔄', config: { n_samples: 5 } },
  { id: 'single', name: 'Single', emoji: '🤖', config: {} },
]

export default function NewWorkflowPage() {
  const router = useRouter()
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [nodes, setNodes] = useState<WorkflowNode[]>([
    { id: 'start', type: 'start', label: 'Start', config: {} },
    { id: 'end', type: 'end', label: 'End', config: {} },
  ])
  const [edges, setEdges] = useState<WorkflowEdge[]>([])
  const [saving, setSaving] = useState(false)

  function addStrategyNode(strategyId: string) {
    const strategy = STRATEGIES.find((s) => s.id === strategyId)
    if (!strategy) return

    const nodeId = `node_${Date.now()}`
    const newNode: WorkflowNode = {
      id: nodeId,
      type: 'strategy',
      label: strategy.name,
      strategy_id: strategy.id,
      config: { ...strategy.config },
    }

    setNodes([...nodes, newNode])
  }

  function removeNode(nodeId: string) {
    // Can't remove start/end
    if (nodeId === 'start' || nodeId === 'end') return

    setNodes(nodes.filter((n) => n.id !== nodeId))
    setEdges(edges.filter((e) => e.from_node_id !== nodeId && e.to_node_id !== nodeId))
  }

  function addEdge(fromId: string, toId: string) {
    const edgeId = `edge_${Date.now()}`
    setEdges([...edges, { id: edgeId, from_node_id: fromId, to_node_id: toId }])
  }

  function removeEdge(edgeId: string) {
    setEdges(edges.filter((e) => e.id !== edgeId))
  }

  async function handleSave() {
    if (!name.trim()) {
      alert('Please enter a workflow name')
      return
    }

    try {
      setSaving(true)
      const workflow = await workflowsApi.create({
        name,
        description,
        org_id: DEFAULT_ORG_ID,
        graph: { nodes, edges },
      })
      router.push(`/workflows/${workflow.id}`)
    } catch (err: any) {
      alert(`Failed to create workflow: ${err.message}`)
    } finally {
      setSaving(false)
    }
  }

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
          <h1 className="text-3xl font-bold text-gray-900">New Workflow</h1>
          <p className="text-gray-600 mt-1">
            Create a workflow using Hidden Layer strategies
          </p>
        </div>
        <button
          onClick={handleSave}
          disabled={saving}
          className="bg-indigo-600 text-white px-6 py-2 rounded-md font-medium hover:bg-indigo-700 transition flex items-center gap-2 disabled:opacity-50"
        >
          <Save size={20} />
          {saving ? 'Saving...' : 'Save Workflow'}
        </button>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Left: Form */}
        <div className="col-span-1 space-y-6">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Workflow Details
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Name *
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="My Workflow"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="What does this workflow do?"
                />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Add Strategy
            </h2>
            <div className="space-y-2">
              {STRATEGIES.map((strategy) => (
                <button
                  key={strategy.id}
                  onClick={() => addStrategyNode(strategy.id)}
                  className="w-full flex items-center gap-3 p-3 border border-gray-200 rounded-lg hover:bg-indigo-50 hover:border-indigo-300 transition text-left"
                >
                  <div className="text-2xl">{strategy.emoji}</div>
                  <div className="flex-1">
                    <div className="font-medium text-gray-900">
                      {strategy.name}
                    </div>
                    <div className="text-xs text-gray-500">{strategy.id}</div>
                  </div>
                  <Plus size={16} className="text-gray-400" />
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Graph builder */}
        <div className="col-span-2 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Workflow Graph
          </h2>

          {/* Nodes */}
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Nodes</h3>
              <div className="space-y-2">
                {nodes.map((node) => (
                  <div
                    key={node.id}
                    className="flex items-center gap-3 p-3 border border-gray-200 rounded-lg bg-gray-50"
                  >
                    <div className="flex-1">
                      <div className="font-medium text-gray-900">
                        {node.label}
                      </div>
                      <div className="text-xs text-gray-500">
                        {node.type}
                        {node.strategy_id && ` • ${node.strategy_id}`}
                        {node.id}
                      </div>
                    </div>
                    {node.type !== 'start' && node.type !== 'end' && (
                      <button
                        onClick={() => removeNode(node.id)}
                        className="p-1 text-gray-400 hover:text-red-600 transition"
                      >
                        <Trash2 size={16} />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Edges */}
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">
                Connections
              </h3>
              <div className="space-y-2">
                {edges.map((edge) => {
                  const fromNode = nodes.find((n) => n.id === edge.from_node_id)
                  const toNode = nodes.find((n) => n.id === edge.to_node_id)
                  return (
                    <div
                      key={edge.id}
                      className="flex items-center gap-3 p-3 border border-gray-200 rounded-lg bg-gray-50"
                    >
                      <div className="flex-1 text-sm">
                        <span className="font-medium">
                          {fromNode?.label || edge.from_node_id}
                        </span>
                        <span className="text-gray-500 mx-2">→</span>
                        <span className="font-medium">
                          {toNode?.label || edge.to_node_id}
                        </span>
                      </div>
                      <button
                        onClick={() => removeEdge(edge.id)}
                        className="p-1 text-gray-400 hover:text-red-600 transition"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  )
                })}
              </div>

              {/* Add edge */}
              <div className="mt-3 p-3 border border-dashed border-gray-300 rounded-lg">
                <div className="text-sm font-medium text-gray-700 mb-2">
                  Add Connection
                </div>
                <div className="flex gap-2">
                  <select
                    onChange={(e) => {
                      const from = e.target.value
                      const to = (e.target.nextElementSibling as HTMLSelectElement)?.value
                      if (from && to && from !== to) {
                        addEdge(from, to)
                        e.target.value = ''
                      }
                    }}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm"
                  >
                    <option value="">From...</option>
                    {nodes.map((n) => (
                      <option key={n.id} value={n.id}>
                        {n.label}
                      </option>
                    ))}
                  </select>
                  <select
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm"
                  >
                    <option value="">To...</option>
                    {nodes.map((n) => (
                      <option key={n.id} value={n.id}>
                        {n.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* Quick create linear workflow */}
          <div className="mt-6 p-4 bg-indigo-50 border border-indigo-200 rounded-lg">
            <div className="text-sm text-indigo-900">
              <strong>Tip:</strong> For a simple workflow, add nodes and connect
              them linearly: start → strategy → end
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
