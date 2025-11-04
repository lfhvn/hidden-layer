# Next Steps - Web Tools

Quick guide to complete the remaining work.

---

## ✅ What's Already Done

**Phase 1 Complete (~60% of total work)**:
- ✅ Full infrastructure (rate limiting, middleware, auth)
- ✅ Multi-Agent Arena MVP (beautiful UI, all strategies working)
- ✅ Steerability Dashboard (gorgeous UI, rate limiting integrated)
- ✅ Comprehensive documentation
- ✅ Import paths fixed
- ✅ Ready for local testing

**What You Can Do Right Now**:
```bash
cd web-tools/multi-agent-arena
cp backend/.env.example backend/.env
# Add ANTHROPIC_API_KEY=sk-ant-... to backend/.env
make setup
make dev
# Visit http://localhost:3001
```

---

## 🎯 Remaining Work

### 1. Add Real-time Streaming (2-3 hours) [OPTIONAL]

**Currently**: Multi-Agent Arena returns full result at end
**Goal**: Stream agent messages as they're generated

**Approach**:

Create `web-tools/multi-agent-arena/backend/app/streaming.py`:

```python
"""Streaming wrapper for multi-agent strategies."""

import asyncio
from typing import AsyncGenerator
from code.strategies import run_strategy

async def stream_debate(question: str, strategy: str, n_agents: int) -> AsyncGenerator[dict, None]:
    """
    Stream debate messages as they're generated.

    Yields messages like:
    {"type": "status", "content": "Starting debate..."}
    {"type": "agent", "agent_id": "agent_1", "content": "I believe..."}
    {"type": "complete", "content": "Final result"}
    """

    # Send initial status
    yield {"type": "status", "content": f"Starting {strategy} with {n_agents} agents..."}

    # For now, we run the full strategy and yield the result
    # TODO: Modify research code to yield intermediate results
    result = run_strategy(
        strategy=strategy,
        task_input=question,
        n_debaters=n_agents,
        provider="anthropic",
        model="claude-3-haiku-20240307"
    )

    # Yield final result
    yield {"type": "complete", "content": result.output}
```

Update WebSocket handler in `main.py`:

```python
from .streaming import stream_debate

@app.websocket("/ws/debate")
async def debate_websocket(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        data = await websocket.receive_json()
        debate_request = DebateRequest(**data)

        # Stream messages
        async for message in stream_debate(
            debate_request.question,
            debate_request.strategy,
            debate_request.n_agents
        ):
            await manager.send_message(AgentMessage(**message), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

Update frontend to use WebSocket:

```typescript
// In useDebateStream.ts hook
const ws = new WebSocket('ws://localhost:8000/ws/debate');

ws.onopen = () => {
  ws.send(JSON.stringify({ question, strategy, n_agents }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  setMessages(prev => [...prev, message]);
};
```

**Estimated Time**: 2-3 hours
**Priority**: Medium (nice-to-have, not required for MVP)

---

### 2. Deploy to Production (1-2 hours) [HIGH PRIORITY]

**Goal**: Get Multi-Agent Arena live on the internet

#### Step 1: Deploy Backend to Railway

```bash
cd web-tools/multi-agent-arena/backend

# Install Railway CLI
npm install -g railway

# Login
railway login

# Initialize project
railway init

# Add environment variables
railway variables set ANTHROPIC_API_KEY=sk-ant-your-key-here
railway variables set CORS_ORIGINS=https://your-frontend.vercel.app
railway variables set RATE_LIMIT_REQUESTS=3
railway variables set RATE_LIMIT_WINDOW=3600

# Deploy
railway up

# Note the URL: https://your-app.railway.app
```

#### Step 2: Deploy Frontend to Vercel

```bash
cd web-tools/multi-agent-arena/frontend

# Install Vercel CLI
npm install -g vercel

# Deploy
vercel

# Add environment variable when prompted
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
NEXT_PUBLIC_WS_URL=wss://your-backend.railway.app

# Deploy to production
vercel --prod

# Note the URL: https://your-app.vercel.app
```

#### Step 3: Update CORS

Go back to Railway and update CORS_ORIGINS:

```bash
railway variables set CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3001
```

#### Step 4: Test Production

1. Visit https://your-app.vercel.app
2. Try a debate
3. Check usage indicator
4. Test rate limiting (make 4 requests quickly)

#### Step 5: Set Up Monitoring (Optional)

**Sentry (Error Tracking)**:
```bash
# Backend
railway variables set SENTRY_DSN=https://...

# Frontend
vercel env add NEXT_PUBLIC_SENTRY_DSN
```

**Cost Alerts**:
- Railway: Dashboard → Settings → Usage → Add Alert at $10
- Anthropic: Dashboard → Usage → Set monthly limit

**Estimated Time**: 1-2 hours
**Priority**: HIGH (get feedback early!)

---

### 3. Build Latent Lens Explorer (1 day) [MEDIUM PRIORITY]

**Goal**: Read-only SAE feature explorer

**Approach**:

#### Step 1: Check Existing Code

```bash
cd projects/latent-space/lens
ls -la
```

#### Step 2: Create Simplified Version

```bash
mkdir -p web-tools/latent-lens
cp -r projects/latent-space/lens/frontend web-tools/latent-lens/
```

#### Step 3: Pre-train Features

```python
# In projects/latent-space/lens
python -m scripts.train_sae --dataset common_crawl --output features.pkl
```

#### Step 4: Create Read-Only API

```python
# web-tools/latent-lens/backend/app/main.py
from fastapi import FastAPI
import pickle

app = FastAPI()

# Load pre-trained features
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

@app.get("/api/features")
async def list_features():
    return {"features": features[:100]}  # Return top 100

@app.post("/api/analyze")
async def analyze_text(text: str):
    # Find which features activate
    activations = compute_activations(text, features)
    return {"activations": activations}
```

#### Step 5: Build Gallery UI

```tsx
// Feature gallery showing discovered features
export function FeatureGallery() {
  return (
    <div className="grid grid-cols-3 gap-4">
      {features.map(feature => (
        <FeatureCard
          id={feature.id}
          description={feature.description}
          examples={feature.activating_examples}
        />
      ))}
    </div>
  )
}
```

**Estimated Time**: 1 day
**Priority**: MEDIUM (interesting but not critical)

---

## 📋 Quick Reference

### Local Development

**Multi-Agent Arena**:
```bash
cd web-tools/multi-agent-arena
make dev
# Frontend: http://localhost:3001
# Backend: http://localhost:8000
```

**Steerability**:
```bash
cd web-tools/steerability
make dev
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

### Common Commands

```bash
# Setup
make setup

# Development
make dev

# Just backend
make dev-backend

# Just frontend
make dev-frontend

# Clean
make clean

# Deploy
make deploy
```

### Environment Variables

**Backend** (`.env`):
```env
ANTHROPIC_API_KEY=sk-ant-...
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
RATE_LIMIT_REQUESTS=5
RATE_LIMIT_WINDOW=3600
ENV=development
LOG_LEVEL=INFO
```

**Frontend** (`.env.local`):
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

---

## 🎯 Recommended Priority Order

### Week 1: Ship MVP

1. ✅ **Test locally** (30 min)
   - Run Multi-Agent Arena
   - Run Steerability
   - Fix any issues

2. 🚀 **Deploy Multi-Agent Arena** (1-2 hours)
   - Railway + Vercel
   - Test in production
   - Set up cost alerts

3. 🚀 **Deploy Steerability** (1-2 hours)
   - Railway + Vercel
   - Test in production

4. 📊 **Monitor & Iterate** (ongoing)
   - Watch usage
   - Gather feedback
   - Fix bugs

### Week 2: Enhance

5. ⚡ **Add Streaming** (2-3 hours)
   - Real-time agent messages
   - Better UX

6. 🔬 **Build Latent Lens** (1 day)
   - Pre-train features
   - Build gallery UI
   - Deploy

### Week 3: Optimize

7. 💰 **Add Caching** (3-4 hours)
   - Redis integration
   - Cache popular queries
   - Reduce API costs 50-80%

8. 🧪 **Add Tests** (1 day)
   - Unit tests
   - Integration tests
   - CI/CD

9. 📈 **Add Monitoring** (2-3 hours)
   - Sentry
   - PostHog
   - Cost dashboard

---

## 💡 Tips

**Cost Management**:
- Start with strict rate limits (3-5 requests/hour)
- Encourage BYOK from day one
- Add caching ASAP
- Monitor costs daily initially

**Getting Feedback**:
- Share on Twitter/HN with demo video
- Ask users what features they want
- Track which strategies are most popular
- See where users drop off

**Iterating**:
- Deploy early, deploy often
- Small incremental improvements
- A/B test different UI approaches
- Let data guide priorities

**Scaling**:
- If usage > 100 sessions/day, add caching immediately
- If costs > $50/day, add Redis
- If costs > $100/day, consider self-hosted inference

---

## 🐛 Troubleshooting

**Import errors**:
```bash
# Make sure you're in project root
cd /home/user/hidden-layer

# Test imports
python -c "from harness import llm_call; print('OK')"
python -c "from code.strategies import run_strategy; print('OK')"
```

**CORS errors**:
```bash
# Check backend .env has correct frontend URL
cat backend/.env | grep CORS_ORIGINS

# Should include: http://localhost:3001 (or your Vercel URL)
```

**Rate limit not working**:
```bash
# Check limiter is in app state
curl http://localhost:8000/api/usage

# Should return: {"limit": 5, "used": 0, "remaining": 5, ...}
```

**WebSocket connection failed**:
```bash
# Check WebSocket URL in frontend
cat frontend/.env.local | grep WS_URL

# Should be: ws://localhost:8000 (or wss://your-backend.railway.app)
```

---

## 📚 Documentation

- **Overview**: `web-tools/README.md`
- **Deployment**: `web-tools/DEPLOYMENT.md`
- **Quick Start**: `web-tools/QUICKSTART.md`
- **Status**: `web-tools/STATUS.md`
- **Progress**: `web-tools/PROGRESS.md`
- **This File**: `web-tools/NEXT_STEPS.md`

---

## ❓ Questions?

- Check `web-tools/PROGRESS.md` for detailed status
- See `web-tools/DEPLOYMENT.md` for deployment help
- Read `web-tools/QUICKSTART.md` for local setup
- Open issue on GitHub if stuck

---

**Remember**: Ship early, iterate based on feedback. The tools are already impressive!

Last Updated: 2025-11-04
