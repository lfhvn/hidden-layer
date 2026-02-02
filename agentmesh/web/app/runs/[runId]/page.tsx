'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { workflowsApi, runsApi, Workflow, WorkflowRun, WorkflowStep } from '@/lib/api'
import { ArrowLeft, PlayCircle, Clock, CheckCircle, XCircle, Loader } from 'lucide-react'

export default function RunDetailPage() {
  const params = useParams()
  const runId = params.runId as string

  const [run, setRun] = useState<WorkflowRun | null>(null)
  const [steps, setSteps] = useState<WorkflowStep[]>([])
  const [workflow, setWorkflow] = useState<Workflow | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadRunData()
    const interval = setInterval(loadRunData, 2000) // Poll every 2s
    return () => clearInterval(interval)
  }, [runId])

  async function loadRunData() {
    try {
      const runData = await runsApi.get(runId)
      setRun(runData)

      const stepsData = await runsApi.getSteps(runId)
      setSteps(stepsData)

      const workflowData = await workflowsApi.get(runData.workflow_id)
      setWorkflow(workflowData)

      setError(null)
    } catch (err: any) {
      setError(err.message || 'Failed to load run')
    } finally {
      setLoading(false)
    }
  }

  if (loading && !run) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading run...</div>
      </div>
    )
  }

  if (error && !run) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="text-red-800 font-medium">Error loading run</div>
        <div className="text-red-600 text-sm mt-1">{error}</div>
      </div>
    )
  }

  if (!run) return null

  const statusIcon = {
    pending: <Clock className="text-gray-500" size={20} />,
    running: <Loader className="text-blue-500 animate-spin" size={20} />,
    succeeded: <CheckCircle className="text-green-500" size={20} />,
    failed: <XCircle className="text-red-500" size={20} />,
    canceled: <XCircle className="text-gray-500" size={20} />,
  }[run.status]

  const statusColor = {
    pending: 'bg-gray-100 text-gray-800',
    running: 'bg-blue-100 text-blue-800',
    succeeded: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
    canceled: 'bg-gray-100 text-gray-800',
  }[run.status]

  const totalCost = steps.reduce((sum, s) => sum + (s.cost_usd || 0), 0)
  const totalTokensIn = steps.reduce((sum, s) => sum + (s.tokens_in || 0), 0)
  const totalTokensOut = steps.reduce((sum, s) => sum + (s.tokens_out || 0), 0)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link
          href={workflow ? `/workflows/${workflow.id}` : '/workflows'}
          className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition"
        >
          <ArrowLeft size={20} />
        </Link>
        <div className="flex-1">
          <h1 className="text-3xl font-bold text-gray-900">Run Details</h1>
          <p className="text-gray-600 mt-1">
            {workflow?.name || 'Loading...'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {statusIcon}
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${statusColor}`}>
            {run.status}
          </span>
        </div>
      </div>

      {/* Metadata */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="text-sm text-gray-500">Run ID</div>
          <div className="text-lg font-mono text-gray-900 truncate" title={run.id}>
            {run.id.slice(0, 8)}...
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="text-sm text-gray-500">Total Cost</div>
          <div className="text-lg font-semibold text-gray-900">
            ${totalCost.toFixed(4)}
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="text-sm text-gray-500">Tokens</div>
          <div className="text-lg font-semibold text-gray-900">
            {totalTokensIn + totalTokensOut}
          </div>
          <div className="text-xs text-gray-500">
            {totalTokensIn} in, {totalTokensOut} out
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="text-sm text-gray-500">Steps</div>
          <div className="text-lg font-semibold text-gray-900">
            {steps.filter((s) => s.status === 'succeeded').length} / {steps.length}
          </div>
          <div className="text-xs text-gray-500">completed</div>
        </div>
      </div>

      {/* Input/Output */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-3">Input</h2>
          <pre className="text-sm bg-gray-50 p-4 rounded border border-gray-200 overflow-auto max-h-64">
            {JSON.stringify(run.input, null, 2)}
          </pre>
        </div>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-3">Output</h2>
          {run.output ? (
            <pre className="text-sm bg-gray-50 p-4 rounded border border-gray-200 overflow-auto max-h-64">
              {JSON.stringify(run.output, null, 2)}
            </pre>
          ) : run.error ? (
            <div className="text-sm text-red-600 bg-red-50 p-4 rounded border border-red-200">
              {run.error}
            </div>
          ) : (
            <div className="text-sm text-gray-500">No output yet...</div>
          )}
        </div>
      </div>

      {/* Steps Timeline */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Execution Timeline
        </h2>
        <div className="space-y-3">
          {steps.map((step, index) => {
            const stepIcon = {
              pending: <Clock className="text-gray-400" size={16} />,
              running: <Loader className="text-blue-500 animate-spin" size={16} />,
              succeeded: <CheckCircle className="text-green-500" size={16} />,
              failed: <XCircle className="text-red-500" size={16} />,
              waiting_human: <Clock className="text-yellow-500" size={16} />,
            }[step.status]

            return (
              <div key={step.id} className="flex gap-4">
                <div className="flex flex-col items-center">
                  <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center">
                    {stepIcon}
                  </div>
                  {index < steps.length - 1 && (
                    <div className="w-0.5 h-12 bg-gray-200 my-1" />
                  )}
                </div>

                <div className="flex-1 pb-4">
                  <div className="flex items-center justify-between">
                    <div className="font-medium text-gray-900">
                      {step.node_id}
                      <span className="ml-2 text-sm text-gray-500">
                        ({step.node_type})
                      </span>
                    </div>
                    <div className="text-sm text-gray-500">
                      {step.latency_s && `${step.latency_s.toFixed(2)}s`}
                    </div>
                  </div>

                  {step.tokens_in && (
                    <div className="text-xs text-gray-500 mt-1">
                      {step.tokens_in} tokens in, {step.tokens_out} tokens out
                      {step.cost_usd && ` • $${step.cost_usd.toFixed(4)}`}
                    </div>
                  )}

                  {step.output && step.status === 'succeeded' && (
                    <details className="mt-2">
                      <summary className="text-sm text-indigo-600 cursor-pointer hover:text-indigo-800">
                        View output
                      </summary>
                      <pre className="text-xs bg-gray-50 p-3 rounded border border-gray-200 mt-2 overflow-auto max-h-32">
                        {typeof step.output === 'string'
                          ? step.output
                          : JSON.stringify(step.output, null, 2)}
                      </pre>
                    </details>
                  )}

                  {step.error && (
                    <div className="text-sm text-red-600 bg-red-50 p-2 rounded border border-red-200 mt-2">
                      {step.error}
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
