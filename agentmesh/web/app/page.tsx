export default function Home() {
  return (
    <div className="space-y-8">
      {/* Hero section */}
      <div className="bg-white rounded-lg shadow-sm p-8 border border-gray-200">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Welcome to AgentMesh
        </h1>
        <p className="text-xl text-gray-600 mb-6">
          Multi-agent workflow orchestration built on Hidden Layer research
        </p>
        <div className="flex gap-4">
          <a
            href="/workflows/new"
            className="bg-indigo-600 text-white px-6 py-3 rounded-md font-medium hover:bg-indigo-700 transition"
          >
            Create Workflow
          </a>
          <a
            href="/workflows"
            className="bg-white text-indigo-600 px-6 py-3 rounded-md font-medium border-2 border-indigo-600 hover:bg-indigo-50 transition"
          >
            View Workflows
          </a>
        </div>
      </div>

      {/* Features */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <div className="text-3xl mb-3">🧠</div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Research-Backed Strategies
          </h3>
          <p className="text-gray-600 text-sm">
            Use proven multi-agent coordination patterns from Hidden Layer research: debate, CRIT, consensus, and more.
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <div className="text-3xl mb-3">⚡</div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Provider Agnostic
          </h3>
          <p className="text-gray-600 text-sm">
            Works with Anthropic Claude, OpenAI GPT, Ollama (local), and MLX. Switch providers instantly.
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <div className="text-3xl mb-3">📊</div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Rich Observability
          </h3>
          <p className="text-gray-600 text-sm">
            Track cost, latency, tokens for every step. Timeline visualization and detailed execution logs.
          </p>
        </div>
      </div>

      {/* Available Strategies */}
      <div className="bg-white rounded-lg shadow-sm p-8 border border-gray-200">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">
          Available Strategies
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { name: 'Debate', desc: 'n-agent debate with judge', emoji: '💬' },
            { name: 'CRIT', desc: 'Multi-perspective design critique', emoji: '🎨' },
            { name: 'Consensus', desc: 'Multiple agents find agreement', emoji: '🤝' },
            { name: 'Manager-Worker', desc: 'Decompose → execute → synthesize', emoji: '👔' },
            { name: 'Self-Consistency', desc: 'Sample multiple times, aggregate', emoji: '🔄' },
            { name: 'Single', desc: 'Baseline single agent', emoji: '🤖' },
          ].map((strategy) => (
            <div key={strategy.name} className="flex items-start p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition">
              <div className="text-2xl mr-3">{strategy.emoji}</div>
              <div>
                <div className="font-semibold text-gray-900">{strategy.name}</div>
                <div className="text-sm text-gray-600">{strategy.desc}</div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-6 p-4 bg-indigo-50 border border-indigo-200 rounded-lg">
          <p className="text-sm text-indigo-900">
            <strong>Research-backed:</strong> All strategies are from Hidden Layer's multi-agent research.
            When research adds new strategies, they're automatically available here.
          </p>
        </div>
      </div>

      {/* Quick Start */}
      <div className="bg-white rounded-lg shadow-sm p-8 border border-gray-200">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Quick Start
        </h2>
        <div className="space-y-4">
          <div className="flex items-start">
            <div className="flex-shrink-0 w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center font-bold text-sm">
              1
            </div>
            <div className="ml-4">
              <div className="font-medium text-gray-900">Create a workflow</div>
              <div className="text-sm text-gray-600">
                Define nodes (strategies) and connect them
              </div>
            </div>
          </div>
          <div className="flex items-start">
            <div className="flex-shrink-0 w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center font-bold text-sm">
              2
            </div>
            <div className="ml-4">
              <div className="font-medium text-gray-900">Execute with input</div>
              <div className="text-sm text-gray-600">
                Provide task input and choose LLM provider
              </div>
            </div>
          </div>
          <div className="flex items-start">
            <div className="flex-shrink-0 w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center font-bold text-sm">
              3
            </div>
            <div className="ml-4">
              <div className="font-medium text-gray-900">View results</div>
              <div className="text-sm text-gray-600">
                See output, timeline, metrics, and cost breakdown
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
