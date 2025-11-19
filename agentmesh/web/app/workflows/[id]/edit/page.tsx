'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import dynamic from 'next/dynamic'
import { workflowsApi, Workflow, WorkflowNode, WorkflowEdge } from '@/lib/api'
import { ArrowLeft, Save, Plus, Trash2 } from 'lucide-react'

// Import ReactFlow dynamically to avoid SSR issues
const WorkflowGraph = dynamic(() => import('@/components/WorkflowGraph'), {
  ssr: false,
  loading: () => <div className="h-[600px] bg-gray-50 rounded-lg border border-gray-200 flex items-center justify-center">Loading graph editor...</div>
})

// Available strategies
const STRATEGIES = [
  { id: 'debate', name: 'Debate', emoji: '💬' },
  { id: 'crit', name: 'CRIT', emoji: '🎨' },
  { id: 'consensus', name: 'Consensus', emoji: '🤝' },
  { id: 'manager_worker', name: 'Manager-Worker', emoji: '👔' },
  { id: 'self_consistency', name: 'Self-Consistency', emoji: '🔄' },
  { id: 'single', name: 'Single', emoji: '🤖' },
]

export default function EditWorkflowPage() {
  const params = useParams()
  const router = useRouter()
  const workflowId = params.id as string

  const [workflow, setWorkflow] = useState<Workflow | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [nodes, setNodes] = useState<WorkflowNode[]>([])
  const [edges, setEdges] = useState<WorkflowEdge[]>([])

  useEffect(() => {
    loadWorkflow()
  }, [workflowId])

  async function loadWorkflow() {
    try {
      setLoading(true)
      const data = await workflowsApi.get(workflowId)
      setWorkflow(data)
      setName(data.name)
      setDescription(data.description || '')
      setNodes(data.graph.nodes)
      setEdges(data.graph.edges)
    } catch (err: any) {
      alert(`Failed to load workflow: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  function addStrategyNode(strategyId: string) {
    const strategy = STRATEGIES.find((s) => s.id === strategyId)
    if (!strategy) return

    const nodeId = `node_${Date.now()}`
    const newNode: WorkflowNode = {
      id: nodeId,
      type: 'strategy',
      label: strategy.name,
      strategy_id: strategy.id,
      config: {},
    }

    setNodes([...nodes, newNode])
  }

  function removeNode(nodeId: string) {
    if (nodeId === 'start' || nodeId === 'end') return
    setNodes(nodes.filter((n) => n.id !== nodeId))
    setEdges(edges.filter((e) => e.from_node_id !== nodeId && e.to_node_id !== nodeId))
  }

  async function handleSave() {
    if (!name.trim()) {
      alert('Please enter a workflow name')
      return
    }

    try {
      setSaving(true)
      // TODO: Implement update workflow API endpoint
      alert('Update endpoint not yet implemented in backend')
      // await workflowsApi.update(workflowId, { name, description, graph: { nodes, edges } })
      // router.push(`/workflows/${workflowId}`)
    } catch (err: any) {
      alert(`Failed to save: ${err.message}`)
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading workflow...</div>
      </div>
    )
  }

  if (!workflow) {
    return <div>Workflow not found</div>
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link
          href={`/workflows/${workflowId}`}
          className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition"
        >
          <ArrowLeft size={20} />
        </Link>
        <div className="flex-1">
          <h1 className="text-3xl font-bold text-gray-900">Edit Workflow</h1>
          <p className="text-gray-600 mt-1">
            Modify workflow structure and settings
          </p>
        </div>
        <button
          onClick={handleSave}
          disabled={saving}
          className="bg-indigo-600 text-white px-6 py-2 rounded-md font-medium hover:bg-indigo-700 transition flex items-center gap-2 disabled:opacity-50"
        >
          <Save size={20} />
          {saving ? 'Saving...' : 'Save Changes'}
        </button>
      </div>

      <div className="grid grid-cols-4 gap-6">
        {/* Left sidebar: Details & Tools */}
        <div className="col-span-1 space-y-4">
          {/* Details */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <h3 className="font-semibold text-gray-900 mb-3">Details</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Name
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
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
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                />
              </div>
            </div>
          </div>

          {/* Add nodes */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <h3 className="font-semibold text-gray-900 mb-3">Add Node</h3>
            <div className="space-y-2">
              {STRATEGIES.map((strategy) => (
                <button
                  key={strategy.id}
                  onClick={() => addStrategyNode(strategy.id)}
                  className="w-full flex items-center gap-2 p-2 border border-gray-200 rounded hover:bg-indigo-50 hover:border-indigo-300 transition text-left text-sm"
                >
                  <span>{strategy.emoji}</span>
                  <span className="flex-1 font-medium">{strategy.name}</span>
                  <Plus size={14} className="text-gray-400" />
                </button>
              ))}
            </div>
          </div>

          {/* Nodes list */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <h3 className="font-semibold text-gray-900 mb-3">Nodes</h3>
            <div className="space-y-1">
              {nodes.map((node) => (
                <div
                  key={node.id}
                  className="flex items-center gap-2 p-2 bg-gray-50 rounded text-sm"
                >
                  <div className="flex-1 truncate">
                    <div className="font-medium">{node.label}</div>
                    <div className="text-xs text-gray-500">{node.type}</div>
                  </div>
                  {node.type !== 'start' && node.type !== 'end' && (
                    <button
                      onClick={() => removeNode(node.id)}
                      className="p-1 text-gray-400 hover:text-red-600"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Graph editor */}
        <div className="col-span-3">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <h3 className="font-semibold text-gray-900 mb-4">Workflow Graph</h3>
            <WorkflowGraph
              nodes={nodes}
              edges={edges}
              onNodesChange={setNodes}
              onEdgesChange={setEdges}
              editable={true}
            />
            <div className="mt-4 text-sm text-gray-600">
              <strong>Tip:</strong> Drag nodes to rearrange. Click and drag from node handles to create connections.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
