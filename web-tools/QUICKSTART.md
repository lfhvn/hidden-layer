# Quick Start Guide

Get a web tool running in 5 minutes.

## Prerequisites

- Docker & Docker Compose (easiest)
- OR Node.js 18+ and Python 3.11+

## Option 1: Docker (Recommended)

### Multi-Agent Arena

```bash
# 1. Navigate to tool
cd web-tools/multi-agent-arena

# 2. Setup environment
cp backend/.env.example backend/.env

# 3. Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" >> backend/.env

# 4. Start everything
docker-compose up

# 5. Open browser
# Frontend: http://localhost:3001
# Backend API: http://localhost:8000/docs
```

### Steerability Dashboard

```bash
cd web-tools/steerability
cp backend/.env.example backend/.env
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" >> backend/.env
docker-compose up

# Frontend: http://localhost:3000
# Backend: http://localhost:8000/docs
```

## Option 2: Local Development (No Docker)

### Multi-Agent Arena

**Terminal 1 - Backend:**
```bash
cd web-tools/multi-agent-arena/backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Edit .env and add your API key
export ANTHROPIC_API_KEY=sk-ant-your-key-here

# Start backend
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd web-tools/multi-agent-arena/frontend
npm install
cp .env.example .env.local

# Edit .env.local if needed (defaults should work)

# Start frontend
npm run dev
```

**Open:** http://localhost:3001

## Getting an Anthropic API Key

1. Go to https://console.anthropic.com/
2. Sign up / log in
3. Go to API Keys
4. Create a new key
5. Copy and paste into `.env` file

**Cost**: ~$0.05-0.30 per multi-agent session with rate limiting enabled

## Testing It Works

### Multi-Agent Arena

1. Open http://localhost:3001
2. Select "Debate" strategy
3. Enter: "Should AI be regulated?"
4. Click "Start Arena"
5. Wait 30-60 seconds
6. See agents debate!

### Steerability

1. Open http://localhost:3000
2. Enter prompt: "Write about the weather"
3. Select steering vector: "Positive Sentiment"
4. Click "Generate"
5. See steered vs unsteered outputs

## Troubleshooting

### "Connection refused" on backend

- Check backend is running on port 8000
- Check CORS settings in backend `.env`
- Make sure API key is set

### "Rate limit exceeded"

- Rate limiting is enabled by default (3 requests/hour)
- Wait or bring your own API key
- Adjust in `.env`: `RATE_LIMIT_REQUESTS=10`

### "Module not found" errors

**Backend:**
```bash
# Make sure you're in the right directory and venv is activated
cd web-tools/multi-agent-arena/backend
source venv/bin/activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd web-tools/multi-agent-arena/frontend
rm -rf node_modules package-lock.json
npm install
```

### Docker issues

```bash
# Clean rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up
```

## Next Steps

- **Customize**: Edit strategies, prompts, UI
- **Deploy**: See `DEPLOYMENT.md` for production deployment
- **Develop**: See `README.md` for development workflows
- **Research**: Explore `/projects/` for underlying research code

## Common Commands

```bash
# Start
make dev                # Docker compose up
make dev-frontend       # Just frontend
make dev-backend        # Just backend

# Stop
docker-compose down     # Stop Docker
Ctrl+C                  # Stop local dev servers

# Clean
make clean              # Remove generated files

# Deploy
make deploy             # Deploy to production
```

## Getting Help

- Check main `README.md`
- See `DEPLOYMENT.md` for hosting
- Open issue on GitHub
- Check backend logs: `docker-compose logs backend`
- Check frontend logs: `docker-compose logs frontend`

## What's Next?

Try these:

1. **Experiment with strategies**: Try debate vs consensus
2. **Test rate limiting**: Make multiple requests
3. **Bring your own key**: Test unlimited mode
4. **Modify UI**: Edit `frontend/src/app/page.tsx`
5. **Add features**: Import from research code

---

**Reminder**: These are demos of research. The core work is in `/projects/` and `/notebooks/`.
