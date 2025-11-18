/**
 * API client for AgentMesh backend.
 */

import axios from 'axios'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Types

export interface WorkflowNode {
  id: string
  type: 'start' | 'end' | 'strategy' | 'tool' | 'human_approval' | 'branch'
  label: string
  config?: Record<string, any>
  strategy_id?: string
  tool_id?: string
  condition?: string
}

export interface WorkflowEdge {
  id: string
  from_node_id: string
  to_node_id: string
  condition?: string
}

export interface WorkflowGraph {
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
}

export interface Workflow {
  id: string
  org_id: string
  name: string
  description?: string
  graph: WorkflowGraph
  created_at: string
  updated_at: string
}

export interface WorkflowRun {
  id: string
  workflow_id: string
  org_id: string
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'canceled'
  input: Record<string, any>
  output?: Record<string, any>
  error?: string
  started_at?: string
  finished_at?: string
  created_at: string
}

export interface WorkflowStep {
  id: string
  run_id: string
  node_id: string
  node_type: string
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'waiting_human'
  input?: any
  output?: any
  error?: string
  latency_s?: number
  tokens_in?: number
  tokens_out?: number
  cost_usd?: number
  started_at?: string
  finished_at?: string
}

// API functions

export const workflowsApi = {
  async list(orgId: string): Promise<Workflow[]> {
    const response = await api.get(`/workflows?org_id=${orgId}`)
    return response.data
  },

  async get(workflowId: string): Promise<Workflow> {
    const response = await api.get(`/workflows/${workflowId}`)
    return response.data
  },

  async create(data: {
    name: string
    description?: string
    org_id: string
    graph: WorkflowGraph
  }): Promise<Workflow> {
    const response = await api.post('/workflows', data)
    return response.data
  },

  async delete(workflowId: string): Promise<void> {
    await api.delete(`/workflows/${workflowId}`)
  },
}

export const runsApi = {
  async create(
    workflowId: string,
    data: {
      input: Record<string, any>
      context?: {
        provider?: string
        model?: string
        temperature?: number
        max_tokens?: number
      }
    }
  ): Promise<WorkflowRun> {
    const response = await api.post(`/workflows/${workflowId}/runs`, data)
    return response.data
  },

  async get(runId: string): Promise<WorkflowRun> {
    const response = await api.get(`/runs/${runId}`)
    return response.data
  },

  async getSteps(runId: string): Promise<WorkflowStep[]> {
    const response = await api.get(`/runs/${runId}/steps`)
    return response.data
  },
}

export default api
