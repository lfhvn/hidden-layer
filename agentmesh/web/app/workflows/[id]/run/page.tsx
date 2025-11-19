'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { workflowsApi, runsApi, Workflow } from '@/lib/api'
import { ArrowLeft, PlayCircle, Loader } from 'lucide-react'

export default function ExecuteWorkflowPage() {
  const params = useParams()
  const router = useRouter()
  const workflowId = params.id as string

  const [workflow, setWorkflow] = useState<Workflow | null>(null)
  const [loading, setLoading] = useState(true)
  const [executing, setExecuting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Form state
  const [taskInput, setTaskInput] = useState('')
  const [provider, setProvider] = useState('anthropic')
  const [model, setModel] = useState('claude-3-5-sonnet-20241022')
  const [temperature, setTemperature] = useState(0.7)

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

  async function handleExecute(e: React.FormEvent) {
    e.preventDefault()

    if (!taskInput.trim()) {
      alert('Please enter a task input')
      return
    }

    try {
      setExecuting(true)
      const run = await runsApi.create(workflowId, {
        input: {
          task: taskInput,
        },
        context: {
          provider,
          model,
          temperature,
        },
      })

      // Redirect to run detail page
      router.push(`/runs/${run.id}`)
    } catch (err: any) {
      alert(`Failed to execute workflow: ${err.message}`)
    } finally {
      setExecuting(false)
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

  const modelOptions = {
    anthropic: [
      { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet' },
      { value: 'claude-3-opus-20240229', label: 'Claude 3 Opus' },
      { value: 'claude-3-haiku-20240307', label: 'Claude 3 Haiku' },
    ],
    openai: [
      { value: 'gpt-4-turbo-preview', label: 'GPT-4 Turbo' },
      { value: 'gpt-4', label: 'GPT-4' },
      { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' },
    ],
    ollama: [
      { value: 'llama3.2:latest', label: 'Llama 3.2' },
      { value: 'mistral:latest', label: 'Mistral' },
      { value: 'codellama:latest', label: 'Code Llama' },
    ],
  }

  const strategyNodes = workflow.graph.nodes.filter((n) => n.type === 'strategy')

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
          <h1 className="text-3xl font-bold text-gray-900">Execute Workflow</h1>
          <p className="text-gray-600 mt-1">{workflow.name}</p>
        </div>
      </div>

      {/* Workflow preview */}
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <div className="text-2xl">⚡</div>
          <div className="flex-1">
            <div className="font-medium text-indigo-900">
              This workflow uses {strategyNodes.length} strategy node
              {strategyNodes.length !== 1 ? 's' : ''}:
            </div>
            <div className="flex gap-2 mt-2 flex-wrap">
              {strategyNodes.map((node) => (
                <span
                  key={node.id}
                  className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800"
                >
                  {node.strategy_id}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Execute form */}
      <form onSubmit={handleExecute} className="space-y-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Workflow Input
          </h2>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Task Input *
            </label>
            <textarea
              value={taskInput}
              onChange={(e) => setTaskInput(e.target.value)}
              rows={6}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder="Enter the task or question for the multi-agent workflow...&#10;&#10;Example: Should we invest in renewable energy? Consider economic, environmental, and technological factors."
              required
            />
            <p className="text-sm text-gray-500 mt-2">
              This will be passed to all strategy nodes in the workflow
            </p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Execution Context
          </h2>

          <div className="space-y-4">
            {/* Provider */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                LLM Provider
              </label>
              <select
                value={provider}
                onChange={(e) => {
                  setProvider(e.target.value)
                  // Reset model when provider changes
                  setModel(modelOptions[e.target.value as keyof typeof modelOptions][0].value)
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="anthropic">Anthropic (Claude)</option>
                <option value="openai">OpenAI (GPT)</option>
                <option value="ollama">Ollama (Local)</option>
              </select>
            </div>

            {/* Model */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model
              </label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {modelOptions[provider as keyof typeof modelOptions].map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Temperature */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Temperature: {temperature}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Focused (0)</span>
                <span>Balanced (0.5)</span>
                <span>Creative (1)</span>
              </div>
            </div>
          </div>

          {/* Cost estimate */}
          <div className="mt-4 p-3 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="text-sm text-gray-700">
              <strong>Estimated cost:</strong> Varies by strategy
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Debate (3 agents) ≈ $0.02-0.05 • Single ≈ $0.005-0.01
            </div>
          </div>
        </div>

        {/* Execute button */}
        <div className="flex justify-end gap-3">
          <Link
            href={`/workflows/${workflowId}`}
            className="px-6 py-2 border border-gray-300 rounded-md font-medium text-gray-700 hover:bg-gray-50 transition"
          >
            Cancel
          </Link>
          <button
            type="submit"
            disabled={executing}
            className="bg-indigo-600 text-white px-8 py-2 rounded-md font-medium hover:bg-indigo-700 transition flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {executing ? (
              <>
                <Loader size={20} className="animate-spin" />
                Executing...
              </>
            ) : (
              <>
                <PlayCircle size={20} />
                Execute Workflow
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  )
}
