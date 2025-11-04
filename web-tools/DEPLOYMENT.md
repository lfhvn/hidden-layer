# Deployment Guide

## Cost Analysis

### Per-Tool Monthly Costs (1000 active users)

| Component | Free Tier | Paid Tier | Notes |
|-----------|-----------|-----------|-------|
| **Frontend (Vercel)** | ✅ Free | $20/mo | Static hosting |
| **Backend (Railway)** | ✅ $5 credit | $10-20/mo | API server |
| **Database (Supabase)** | ✅ Free | $25/mo | If needed |
| **API Costs (Anthropic)** | - | $50-500/mo | Depends on usage |

### Total: $10-50/month + API usage

### Cost Mitigation Strategies

1. **Rate Limiting**: 5 requests/hour for free users
2. **Caching**: Cache popular queries (reduce API calls by 50-80%)
3. **Smaller Models**: Offer Haiku for free tier, Sonnet for paid
4. **BYOK**: Users bring their own API keys (unlimited usage)

## Deployment Options

### Option 1: Full Managed (Recommended for Start)

**Frontend**: Vercel
- Free tier: 100GB bandwidth
- Auto SSL, CDN, preview deployments
- Deploy: `vercel deploy`

**Backend**: Railway
- $5 free credit/month
- Easy scaling
- Deploy: `railway up`

**Pros**: Easy, reliable, good free tier
**Cons**: Less control, costs scale with usage

### Option 2: Self-Hosted (For Growth)

**Frontend**: Cloudflare Pages (Free)
**Backend**: Your own VPS ($5-20/mo)
**Database**: PostgreSQL on same VPS

**Pros**: Fixed costs, full control
**Cons**: More maintenance

### Option 3: Hybrid

**Frontend**: Vercel (free)
**Backend**: Cloud Run (pay-per-use)

**Pros**: Best of both worlds
**Cons**: More complex setup

## Quick Deploy: Steerability Dashboard

### 1. Deploy Backend (Railway)

```bash
cd web-tools/steerability/backend

# Install Railway CLI
npm install -g railway

# Login and deploy
railway login
railway init
railway up

# Set environment variables
railway variables set ANTHROPIC_API_KEY=sk-...
railway variables set CORS_ORIGINS=https://your-frontend.vercel.app
```

### 2. Deploy Frontend (Vercel)

```bash
cd web-tools/steerability/frontend

# Install Vercel CLI
npm install -g vercel

# Deploy
vercel

# Set environment variable
vercel env add NEXT_PUBLIC_API_URL
# Enter: https://your-backend.railway.app
```

### 3. Configure Rate Limiting

Edit backend `.env`:
```env
RATE_LIMIT_REQUESTS=5
RATE_LIMIT_WINDOW=3600  # 1 hour
ENABLE_API_KEY_MODE=true
```

### 4. Monitor Costs

Set up billing alerts:
- Railway: Settings → Usage → Set alert at $10
- Vercel: Settings → Billing → Set alert
- Anthropic: Dashboard → Usage → Set limit

## Environment Variables

### Backend (.env)

```env
# Required
ANTHROPIC_API_KEY=sk-ant-...
CORS_ORIGINS=https://your-frontend.vercel.app,http://localhost:3000

# Rate Limiting
RATE_LIMIT_REQUESTS=5
RATE_LIMIT_WINDOW=3600
ENABLE_API_KEY_MODE=true

# Model Selection
DEFAULT_MODEL=claude-3-haiku-20240307
ALLOW_MODEL_SELECTION=true

# Caching
ENABLE_CACHE=true
CACHE_TTL=3600

# Monitoring (optional)
SENTRY_DSN=https://...
POSTHOG_KEY=phc_...
```

### Frontend (.env.local)

```env
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
NEXT_PUBLIC_POSTHOG_KEY=phc_...  # Optional
```

## Rate Limiting Configuration

### Simple IP-based (Default)

```python
# backend/app/middleware/rate_limit.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/generate")
@limiter.limit("5/hour")
async def generate(request: Request):
    ...
```

### API Key-based (Unlimited)

```python
@app.post("/api/generate")
async def generate(request: Request, api_key: str = Header(None)):
    if api_key:
        # User's own key, no rate limit
        return await generate_with_key(api_key)
    else:
        # Use our key, apply rate limit
        await rate_limiter.check(request)
        return await generate_with_our_key()
```

## Monitoring Setup

### 1. Error Tracking (Sentry)

```bash
# Backend
pip install sentry-sdk[fastapi]

# In app/main.py
import sentry_sdk
sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"))
```

```bash
# Frontend
npm install @sentry/nextjs

# Automatically configured
npx @sentry/wizard -i nextjs
```

### 2. Usage Analytics (PostHog)

```bash
# Frontend
npm install posthog-js

# In app/layout.tsx
import { PostHogProvider } from 'posthog-js/react'
```

### 3. Cost Alerts

**Railway**:
- Dashboard → Settings → Usage → Add Alert
- Alert at: $10, $20, $50

**Anthropic**:
- Dashboard → Usage → Set monthly limit
- Alert at 80% of limit

## Scaling Strategy

### Phase 1: Launch (0-100 users)
- Free tiers
- Simple rate limiting
- Our API keys
- **Cost**: $0-10/month

### Phase 2: Growth (100-1000 users)
- Add caching
- Offer BYOK option
- Paid tier ($5/mo unlimited)
- **Cost**: $50-100/month

### Phase 3: Scale (1000+ users)
- Move to self-hosted backend
- Implement job queue for long tasks
- Multiple model providers
- **Cost**: $200-500/month

## Troubleshooting

### High API Costs

1. Enable caching: `ENABLE_CACHE=true`
2. Use smaller models: `DEFAULT_MODEL=claude-3-haiku-20240307`
3. Tighten rate limits: `RATE_LIMIT_REQUESTS=3`
4. Require API keys: `ENABLE_API_KEY_MODE=true`

### Backend Timeouts

1. Increase timeout: `TIMEOUT=60`
2. Add streaming: `stream=True` in API calls
3. Use websockets for long operations

### CORS Issues

Add to backend `.env`:
```env
CORS_ORIGINS=https://your-frontend.vercel.app,https://your-frontend.com,http://localhost:3000
```

## Security Checklist

- [ ] Rate limiting enabled
- [ ] CORS configured (not *)
- [ ] API keys in environment variables (not committed)
- [ ] HTTPS enabled (automatic with Vercel/Railway)
- [ ] Input validation on all endpoints
- [ ] Sentry error tracking enabled
- [ ] Cost alerts configured

## Maintenance

### Weekly
- Check error logs (Sentry)
- Review usage (PostHog)
- Check costs (Railway/Vercel)

### Monthly
- Review rate limits (adjust if needed)
- Check cache hit rate
- Review model usage (optimize for cost)
- Update dependencies

## Questions?

See main `README.md` or open an issue.
