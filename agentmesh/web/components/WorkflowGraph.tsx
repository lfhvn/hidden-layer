'use client'

import { useCallback } from 'react'
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  MarkerType,
  NodeTypes,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { WorkflowNode as WorkflowNodeType, WorkflowEdge as WorkflowEdgeType } from '@/lib/api'

interface WorkflowGraphProps {
  nodes: WorkflowNodeType[]
  edges: WorkflowEdgeType[]
  onNodesChange?: (nodes: WorkflowNodeType[]) => void
  onEdgesChange?: (edges: WorkflowEdgeType[]) => void
  editable?: boolean
}

// Custom node component
function CustomNode({ data }: { data: any }) {
  const bgColor = {
    start: 'bg-green-100 border-green-500',
    end: 'bg-red-100 border-red-500',
    strategy: 'bg-indigo-100 border-indigo-500',
    tool: 'bg-yellow-100 border-yellow-500',
    human_approval: 'bg-purple-100 border-purple-500',
    branch: 'bg-orange-100 border-orange-500',
  }[data.type] || 'bg-gray-100 border-gray-500'

  const icon = {
    start: '▶️',
    end: '⏹️',
    strategy: '⚡',
    tool: '🔧',
    human_approval: '👤',
    branch: '🔀',
  }[data.type] || '●'

  return (
    <div className={`px-4 py-3 shadow-md rounded-lg border-2 ${bgColor} min-w-[150px]`}>
      <div className="flex items-center gap-2">
        <div className="text-lg">{icon}</div>
        <div className="flex-1">
          <div className="font-semibold text-gray-900">{data.label}</div>
          {data.strategy_id && (
            <div className="text-xs text-gray-600 mt-1">
              {data.strategy_id}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const nodeTypes: NodeTypes = {
  custom: CustomNode,
}

export default function WorkflowGraph({
  nodes: workflowNodes,
  edges: workflowEdges,
  onNodesChange,
  onEdgesChange,
  editable = false,
}: WorkflowGraphProps) {
  // Convert workflow nodes to ReactFlow nodes
  const initialNodes: Node[] = workflowNodes.map((node, index) => ({
    id: node.id,
    type: 'custom',
    position: { x: 250, y: index * 100 }, // Simple vertical layout
    data: {
      label: node.label,
      type: node.type,
      strategy_id: node.strategy_id,
    },
  }))

  // Convert workflow edges to ReactFlow edges
  const initialEdges: Edge[] = workflowEdges.map((edge) => ({
    id: edge.id,
    source: edge.from_node_id,
    target: edge.to_node_id,
    type: 'smoothstep',
    animated: true,
    markerEnd: {
      type: MarkerType.ArrowClosed,
    },
  }))

  const [nodes, setNodes, onNodesChangeInternal] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChangeInternal] = useEdgesState(initialEdges)

  const onConnect = useCallback(
    (params: Connection) => {
      if (!editable) return

      const newEdges = addEdge(params, edges)
      setEdges(newEdges)

      // Notify parent component
      if (onEdgesChange) {
        const workflowEdges: WorkflowEdgeType[] = newEdges.map((e) => ({
          id: e.id,
          from_node_id: e.source,
          to_node_id: e.target,
        }))
        onEdgesChange(workflowEdges)
      }
    },
    [edges, editable, onEdgesChange]
  )

  return (
    <div className="w-full h-[600px] bg-gray-50 rounded-lg border border-gray-200">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={editable ? onNodesChangeInternal : undefined}
        onEdgesChange={editable ? onEdgesChangeInternal : undefined}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  )
}
