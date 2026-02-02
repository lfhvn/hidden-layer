# Multi-Agent Arena - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User (Mobile/Desktop)                    │
│              https://multi-agent.vercel.app                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ HTTPS/WSS
                         │
┌────────────────────────▼────────────────────────────────────┐
│                 Frontend (Next.js)                          │
│                   Hosted on Vercel                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Components                                           │  │
│  │ • StrategySelector (choose debate/consensus/etc)    │  │
│  │ • QuestionInput (enter question)                    │  │
│  │ • StreamingDebateViewer (real-time updates)        │  │
│  │ • UsageIndicator (rate limit tracking)             │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ API Calls (HTTPS)
                         │ WebSocket (WSS)
                         │
┌────────────────────────▼────────────────────────────────────┐
│               Backend (FastAPI)                             │
│            Hosted on Railway                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ API Endpoints                                        │  │
│  │ • GET  /                 (health check)             │  │
│  │ • GET  /api/strategies   (list available)          │  │
│  │ • GET  /api/usage        (rate limit status)       │  │
│  │ • POST /api/debate       (run debate)              │  │
│  │ • WS   /ws/debate        (streaming)               │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Middleware                                           │  │
│  │ • CORS (frontend URL whitelist)                     │  │
│  │ • Rate Limiting (3 requests/hour/IP)                │  │
│  │ • API Key Validation (BYOK support)                 │  │
│  │ • Error Handling                                    │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Import
                         │
┌────────────────────────▼────────────────────────────────────┐
│           Research Code (Multi-Agent)                       │
│       /communication/multi-agent/multi_agent/               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Strategies (strategies.py)                           │  │
│  │ • Debate        (3+ agents + judge)                  │  │
│  │ • Consensus     (collaborative)                      │  │
│  │ • CRIT          (design critique)                    │  │
│  │ • Manager-Worker (decompose + synthesize)            │  │
│  │ • Self-Consistency (sample + aggregate)              │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ LLM Calls
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Harness (LLM Provider)                     │
│                      /harness/                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Provider Abstraction                                 │  │
│  │ • Anthropic API (Claude)                            │  │
│  │ • OpenAI API (GPT) - optional                       │  │
│  │ • Ollama (local) - optional                         │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ API Calls
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Anthropic Claude API                           │
│           (claude-3-haiku-20240307)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. User Initiates Debate

```
User → Frontend → Backend
```

**User Actions**:
1. Selects strategy (e.g., "Debate")
2. Enters question (e.g., "Should AI be regulated?")
3. Clicks "Start Debate"

**Frontend**:
1. Validates input
2. Checks rate limit status
3. Opens WebSocket connection to backend
4. Sends request: `{question, strategy, n_agents}`

### 2. Backend Processes Request

```
Backend → Rate Limiter → Strategy Executor → LLM Provider
```

**Rate Limiter**:
- Checks IP address
- Verifies: requests < 3 in last hour
- Rejects if limit exceeded
- OR allows if valid API key provided (BYOK)

**Strategy Executor**:
- Imports `run_strategy` from research code
- Calls with: `(strategy, question, n_agents, provider, model)`
- Returns structured result

### 3. Research Code Executes Strategy

```
Strategy → Multiple LLM Calls → Synthesis
```

**Example: Debate Strategy**
1. **Agent 1**: Generate "pro" position
2. **Agent 2**: Generate "con" position
3. **Agent 3**: Generate "neutral" position
4. **Judge**: Synthesize all positions → final answer

Each step calls `harness.llm_call(prompt, provider, model)`

### 4. Response Streams Back

```
LLM → Harness → Backend → WebSocket → Frontend → User
```

**Streaming Process**:
- Backend yields messages as strategy executes
- WebSocket sends each message to frontend
- Frontend updates UI in real-time
- Types: "status", "agent", "judge", "complete"

**Message Format**:
```json
{
  "type": "agent",
  "agent_id": "agent_1",
  "role": "Supporter",
  "content": "I believe regulation is essential because..."
}
```

---

## Component Details

### Frontend (Next.js + React)

**Tech Stack**:
- Next.js 14 (React framework)
- TypeScript (type safety)
- Tailwind CSS (styling)
- WebSocket API (real-time)

**Key Files**:
- `src/app/page.tsx` - Main arena UI
- `src/components/StreamingDebateViewer.tsx` - Real-time debate display
- `src/components/StrategySelector.tsx` - Strategy picker
- `src/hooks/useDebateStream.ts` - WebSocket hook

**State Management**:
- React hooks (useState, useEffect)
- WebSocket connection state
- Debate messages array
- Rate limit tracking

### Backend (FastAPI + Python)

**Tech Stack**:
- FastAPI (async web framework)
- Uvicorn (ASGI server)
- WebSockets (real-time)
- Pydantic (validation)

**Key Files**:
- `app/main.py` - FastAPI app, endpoints, WebSocket
- `app/streaming.py` - Strategy streaming wrapper
- `requirements.txt` - Dependencies

**Middleware Stack**:
1. CORS (cross-origin)
2. Rate Limiting (IP-based)
3. API Key Validation (BYOK)
4. Error Handling
5. Logging

### Research Code (Multi-Agent Strategies)

**Location**: `/communication/multi-agent/multi_agent/`

**Key Modules**:
- `strategies.py` - Strategy implementations (1131 lines)
- `debate.py` - Debate-specific logic
- `consensus.py` - Consensus building
- `crit/strategies.py` - Design critique
- `manager_worker.py` - Decomposition strategy

**Strategy Pattern**:
```python
def run_strategy(
    strategy: str,
    task_input: str,
    n_debaters: int,
    provider: str,
    model: str
) -> StrategyResult
```

All strategies return consistent format for easy web integration.

### Harness (LLM Provider Abstraction)

**Location**: `/harness/`

**Purpose**: Unified interface for all LLM providers

**Key Functions**:
```python
from harness import llm_call

response = llm_call(
    prompt="Your question here",
    provider="anthropic",  # or "openai", "ollama"
    model="claude-3-haiku-20240307"
)
```

**Benefits**:
- Switch providers without changing strategy code
- Consistent error handling
- Experiment tracking
- Cost monitoring

---

## Deployment Architecture

### Production Setup

```
┌─────────────────────────────────────────────────────────┐
│                    GitHub Repository                    │
│                   lfhvn/hidden-layer                    │
└──────────────┬────────────────────────┬─────────────────┘
               │                        │
               │ Auto-deploy            │ Auto-deploy
               │ on push                │ on push
               ▼                        ▼
┌──────────────────────────┐  ┌───────────────────────────┐
│   Railway (Backend)      │  │   Vercel (Frontend)       │
│   • Python 3.11          │  │   • Node.js 20            │
│   • Uvicorn server       │  │   • Next.js 14            │
│   • Auto-scaling         │  │   • Edge network          │
│   • Environment vars     │  │   • Global CDN            │
│   • Logs & monitoring    │  │   • Analytics             │
└──────────────────────────┘  └───────────────────────────┘
```

### Environment Variables Flow

**Development** (`.env.local`):
```
Frontend: NEXT_PUBLIC_API_URL=http://localhost:8000
Backend:  ANTHROPIC_API_KEY=sk-ant-...
```

**Production**:
```
Frontend (Vercel): NEXT_PUBLIC_API_URL=https://xxx.railway.app
Backend (Railway): ANTHROPIC_API_KEY=sk-ant-...
                   CORS_ORIGINS=https://xxx.vercel.app
```

---

## Security Architecture

### Rate Limiting

**Implementation**:
```python
class RateLimiter:
    def check(self, ip_address: str) -> bool:
        key = f"rate_limit:{ip_address}"
        count = redis.incr(key)
        if count == 1:
            redis.expire(key, window)
        return count <= limit
```

**Default Settings**:
- 3 requests per hour per IP
- 1 hour reset window
- Bypass with valid API key (BYOK)

### CORS Protection

**Whitelist**:
- Production frontend URL only
- Localhost for development
- No wildcards

**Headers**:
- `Access-Control-Allow-Origin`: Specific domains only
- `Access-Control-Allow-Credentials`: false (no cookies)
- `Access-Control-Allow-Methods`: GET, POST, OPTIONS

### API Key Security

**Storage**:
- Never in code
- Environment variables only
- Railway/Vercel secrets

**Usage**:
- Backend uses lab API key (rate limited)
- Users can provide own key (BYOK)
- Keys validated before LLM calls

---

## Performance Characteristics

### Latency

**Typical Request**:
1. Frontend → Backend: ~50-100ms (network)
2. Rate limit check: ~1-5ms (in-memory)
3. Strategy execution: 3-15 seconds (LLM calls)
   - Debate (3 agents): ~10 seconds
   - Consensus: ~8 seconds
   - Manager-Worker: ~12 seconds
4. Response streaming: Real-time (WebSocket)

**Optimization**:
- WebSocket keeps connection open
- Streaming updates (no waiting for completion)
- Async processing (parallel LLM calls where possible)

### Scalability

**Frontend (Vercel)**:
- Automatic edge caching
- Global CDN distribution
- Serverless functions
- Scales to millions of users

**Backend (Railway)**:
- Auto-scaling based on load
- Resource limits configurable
- Horizontal scaling available

**Bottleneck**: LLM API calls
- Limited by Anthropic rate limits
- Rate limiting prevents abuse
- Caching can reduce duplicate calls

### Cost Structure

**Per Debate**:
- 3 agents × ~500 tokens = 1500 tokens input
- 3 responses × ~300 tokens = 900 tokens output
- Haiku cost: ~$0.05-0.10 per debate

**Monthly (100 users/week, 3 debates/hour)**:
- API costs: $20-50
- Hosting: Free tier sufficient
- Total: ~$20-50/month

---

## Monitoring & Observability

### Metrics to Track

**Railway (Backend)**:
- Request rate
- Error rate
- Response time (p50, p95, p99)
- Memory usage
- CPU usage

**Vercel (Frontend)**:
- Page views
- Load time
- Bandwidth usage
- Function executions

**Application**:
- Debates per hour
- Strategy distribution
- Average debate length
- User retention

### Logging

**Backend Logs**:
```
2025-11-10 19:16:57 INFO - Started debate: strategy=debate, n_agents=3
2025-11-10 19:16:58 INFO - Agent 1 response: 245 tokens
2025-11-10 19:16:59 INFO - Agent 2 response: 198 tokens
2025-11-10 19:17:01 INFO - Judge synthesis: 312 tokens
2025-11-10 19:17:01 INFO - Debate complete: total_time=4.2s
```

**Frontend Logs**:
- Console errors
- WebSocket connection status
- API call failures

---

## Future Architecture Enhancements

### Planned Improvements

1. **Caching Layer**
   - Redis for popular questions
   - Reduce API costs by 30-50%
   - TTL: 24 hours

2. **User Accounts**
   - Authentication (Clerk/Auth0)
   - Personal API keys
   - Usage tracking per user
   - Favorite strategies

3. **Database**
   - Supabase/PostgreSQL
   - Store debate history
   - Analytics
   - Leaderboards

4. **Enhanced Streaming**
   - Token-by-token streaming
   - Progress indicators
   - Partial results

5. **Mobile App**
   - React Native version
   - Offline mode
   - Push notifications
   - Share debates

---

## Development vs Production

### Development

**Backend**:
```bash
uvicorn app.main:app --reload --port 8000
```
- Hot reload enabled
- Verbose logging
- No rate limiting
- Local database

**Frontend**:
```bash
npm run dev
```
- Fast refresh
- Source maps
- Development build
- CORS: localhost

### Production

**Backend (Railway)**:
- Optimized Python build
- Production logging
- Rate limiting active
- Environment-based config

**Frontend (Vercel)**:
- Minified build
- Image optimization
- Edge caching
- Analytics enabled

---

## Questions?

See also:
- `DEPLOYMENT_GUIDE.md` - Full deployment instructions
- `QUICK_DEPLOY.md` - 15-minute quick start
- `README.md` - Project overview
