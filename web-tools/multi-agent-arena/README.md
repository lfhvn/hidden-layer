# 🤝 Multi-Agent Arena

Watch AI agents debate, critique, and solve problems in real-time.

## Features

- **Live Debates**: Watch 3+ agents debate complex topics
- **Multiple Strategies**: Debate, consensus, design critique, manager-worker
- **Real-time Streaming**: See each agent's thoughts as they think
- **Strategy Comparison**: Side-by-side results
- **Vote & Share**: Community can vote on best strategies

## Quick Start

```bash
# Development
cd web-tools/multi-agent-arena
cp .env.example .env
make dev

# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

## Usage

### 1. Choose a Strategy

- **Debate**: Agents argue different perspectives, judge decides
- **Consensus**: Agents work together to find agreement
- **Design Critique** (CRIT): Multiple perspectives critique a design
- **Manager-Worker**: Manager decomposes, workers solve, synthesis

### 2. Enter Your Question

Example prompts:
- "Should we invest in renewable energy?"
- "What's the best approach to teaching kids about AI?"
- "How can we reduce urban traffic congestion?"

### 3. Watch the Magic

See agents:
- Present their perspectives
- Respond to each other
- Build on ideas
- Reach conclusions

### 4. Compare Strategies

Run the same prompt with different strategies and see which produces better results.

## Architecture

### Backend (FastAPI + WebSockets)

```
backend/
├── app/
│   ├── main.py              # FastAPI app
│   ├── strategies/          # Strategy implementations
│   │   ├── debate.py
│   │   ├── consensus.py
│   │   ├── crit.py
│   │   └── manager_worker.py
│   ├── streaming.py         # WebSocket streaming
│   └── models.py            # Pydantic models
└── requirements.txt
```

### Frontend (Next.js + React)

```
frontend/
├── src/
│   ├── app/
│   │   ├── page.tsx         # Main arena
│   │   └── compare/         # Strategy comparison
│   ├── components/
│   │   ├── AgentMessage.tsx
│   │   ├── StrategySelector.tsx
│   │   └── DebateViewer.tsx
│   └── hooks/
│       └── useDebateStream.ts
└── package.json
```

## Strategies

### Debate (3+ agents + judge)

```
User Question
  ↓
Agent 1 (Position A) ←→ Agent 2 (Position B) ←→ Agent 3 (Position C)
  ↓
Judge synthesizes and decides
```

**Best for**: Controversial topics, multiple valid viewpoints

### Consensus (3+ agents)

```
User Question
  ↓
Agent 1 → Proposal
Agent 2 → Builds on proposal
Agent 3 → Refines
  ↓
Agreement reached
```

**Best for**: Finding common ground, collaborative solutions

### CRIT (Designer + Reviewers)

```
Design Submission
  ↓
UX Reviewer  →  Technical Reviewer  →  Business Reviewer
  ↓
Synthesis of feedback
```

**Best for**: Design/product critique, structured feedback

### Manager-Worker

```
Complex Problem
  ↓
Manager: Decomposes into subtasks
  ↓
Worker 1    Worker 2    Worker 3
  ↓
Manager: Synthesizes solutions
```

**Best for**: Complex problems, parallel decomposition

## Configuration

### Environment Variables

```env
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Rate Limiting
RATE_LIMIT_REQUESTS=3
RATE_LIMIT_WINDOW=3600

# Strategy Config
MAX_AGENTS=5
MAX_ROUNDS=3
ENABLE_STREAMING=true

# Model Selection
DEFAULT_MODEL=claude-3-haiku-20240307
ALLOW_MODEL_SELECTION=false  # Only haiku for free tier
```

### Cost Controls

**Critical**: Multi-agent is expensive! Each session makes 6-10 API calls.

- **Rate Limit**: 3 sessions per hour (default)
- **Max Agents**: Limited to 5
- **Max Rounds**: Limited to 3
- **Model**: Haiku only for free tier
- **Cache**: Cache popular prompts

Estimated cost per session: $0.05-0.30

## Development

### Run Locally

```bash
# Start backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Start frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Test Strategies

```python
# Test debate strategy
curl -X POST http://localhost:8000/api/debate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Should AI be regulated?",
    "n_agents": 3
  }'
```

### WebSocket Testing

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/debate');

ws.send(JSON.stringify({
  question: "Should AI be regulated?",
  strategy: "debate",
  n_agents: 3
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Agent ${data.agent}: ${data.message}`);
};
```

## Deployment

See `../DEPLOYMENT.md` for full deployment guide.

### Quick Deploy

```bash
# Deploy backend (Railway)
railway up

# Deploy frontend (Vercel)
vercel deploy

# Set environment variables
railway variables set ANTHROPIC_API_KEY=sk-...
vercel env add NEXT_PUBLIC_WS_URL
```

## Monitoring

Track these metrics:
- Sessions per hour
- Average cost per session
- Strategy popularity
- Error rate
- User satisfaction (votes)

## Limitations

- **Free Tier**: 3 debates per hour
- **Max Agents**: 5 agents per debate
- **Max Rounds**: 3 rounds per debate
- **Model**: Haiku only (upgrade for Sonnet)
- **Timeout**: 2 minutes per session

## Future Ideas

- [ ] Agent personas (e.g., "optimist", "skeptic", "pragmatist")
- [ ] Tournament mode (agents compete)
- [ ] Community voting on best responses
- [ ] Save and share debate transcripts
- [ ] Custom system prompts
- [ ] Multi-language support

## Research Connection

This tool showcases research from `/projects/multi-agent/`.

**Research Questions**:
- When do multi-agent strategies outperform single models?
- What coordination mechanisms emerge?
- How does strategy choice affect output quality?

**Contribute**: Try strategies, share interesting examples, suggest improvements!

## Troubleshooting

### WebSocket Connection Failed

Check CORS settings in backend `.env`:
```env
CORS_ORIGINS=http://localhost:3000,https://your-frontend.vercel.app
```

### High Costs

1. Reduce rate limit: `RATE_LIMIT_REQUESTS=2`
2. Lower max agents: `MAX_AGENTS=3`
3. Lower max rounds: `MAX_ROUNDS=2`
4. Enable aggressive caching

### Slow Responses

1. Enable streaming: `ENABLE_STREAMING=true`
2. Use parallel API calls where possible
3. Reduce max rounds

## Questions?

See main `../README.md` or open an issue.
