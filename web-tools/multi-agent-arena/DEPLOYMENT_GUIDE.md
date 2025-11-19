# Multi-Agent Arena - Deployment Guide

Complete step-by-step guide to deploy the Multi-Agent Arena for remote access.

## Prerequisites

Before starting, you'll need:

1. **Anthropic API Key**
   - Get one at: https://console.anthropic.com/
   - Navigate to API Keys → Create Key
   - Copy and save securely

2. **Railway Account** (for backend)
   - Sign up at: https://railway.app/
   - Free tier: $5 credit (enough for testing)
   - GitHub authentication recommended

3. **Vercel Account** (for frontend)
   - Sign up at: https://vercel.com/
   - Free tier: Unlimited deployments
   - GitHub authentication recommended

---

## Option 1: Deploy via Web Dashboards (Recommended for First Time)

### Step 1: Deploy Backend to Railway

#### 1.1 Create New Project

1. Go to https://railway.app/dashboard
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Authorize Railway to access your GitHub
5. Select repository: `lfhvn/hidden-layer`
6. Click **"Add variables"** before deploying

#### 1.2 Configure Build Settings

Railway should auto-detect Python, but verify:

- **Root Directory**: `web-tools/multi-agent-arena/backend`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

#### 1.3 Set Environment Variables

Click **"Variables"** tab and add:

```
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
CORS_ORIGINS=http://localhost:3001
RATE_LIMIT_REQUESTS=3
RATE_LIMIT_WINDOW=3600
MAX_AGENTS=5
MAX_ROUNDS=3
DEFAULT_MODEL=claude-3-haiku-20240307
ALLOW_MODEL_SELECTION=false
ENV=production
LOG_LEVEL=INFO
```

**Important**: We'll update `CORS_ORIGINS` after deploying frontend.

#### 1.4 Deploy

1. Click **"Deploy"**
2. Wait for build to complete (~2-3 minutes)
3. Railway will assign a URL like: `https://multi-agent-backend.up.railway.app`
4. **Save this URL** - you'll need it for frontend setup

#### 1.5 Verify Backend

Test your backend:

```bash
curl https://your-backend-url.railway.app/
# Should return: {"status":"ok","service":"multi-agent-arena","version":"0.1.0"}

curl https://your-backend-url.railway.app/api/strategies
# Should return list of strategies
```

---

### Step 2: Deploy Frontend to Vercel

#### 2.1 Create New Project

1. Go to https://vercel.com/dashboard
2. Click **"Add New..."** → **"Project"**
3. Import `lfhvn/hidden-layer` repository
4. Configure project:
   - **Framework Preset**: Next.js
   - **Root Directory**: `web-tools/multi-agent-arena/frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`

#### 2.2 Set Environment Variables

Before deploying, add environment variables:

```
NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
NEXT_PUBLIC_WS_URL=wss://your-backend-url.railway.app
```

Replace `your-backend-url.railway.app` with the URL from Step 1.4.

#### 2.3 Deploy

1. Click **"Deploy"**
2. Wait for build (~2-3 minutes)
3. Vercel will assign a URL like: `https://multi-agent-arena.vercel.app`
4. **Save this URL**

#### 2.4 Update Backend CORS

Now that you have the frontend URL, update Railway backend:

1. Go back to Railway dashboard
2. Select your backend project
3. Go to **"Variables"**
4. Update `CORS_ORIGINS`:
   ```
   CORS_ORIGINS=https://multi-agent-arena.vercel.app,http://localhost:3001
   ```
5. Click **"Save"** - Railway will auto-redeploy

---

### Step 3: Test Deployment

1. Visit your Vercel URL: `https://multi-agent-arena.vercel.app`
2. You should see the Multi-Agent Arena interface
3. Try a test debate:
   - Select strategy: **Debate**
   - Enter question: "Should AI be regulated?"
   - Click **"Start Debate"**
4. Watch agents respond in real-time

**If you see errors**, check:
- Browser console (F12) for frontend errors
- Railway logs for backend errors
- CORS configuration is correct
- Environment variables are set

---

## Option 2: Deploy via CLI (Faster for Repeat Deployments)

### Install CLI Tools

```bash
# Install Railway CLI
npm install -g @railway/cli

# Install Vercel CLI
npm install -g vercel
```

### Deploy Backend (Railway)

```bash
cd web-tools/multi-agent-arena/backend

# Login
railway login

# Create new project (first time only)
railway init

# Set environment variables
railway variables set ANTHROPIC_API_KEY="sk-ant-..."
railway variables set CORS_ORIGINS="http://localhost:3001"
railway variables set RATE_LIMIT_REQUESTS=3
railway variables set RATE_LIMIT_WINDOW=3600

# Deploy
railway up

# Get URL
railway domain
```

### Deploy Frontend (Vercel)

```bash
cd web-tools/multi-agent-arena/frontend

# Login
vercel login

# Deploy to production
vercel --prod

# Set environment variables (first time only)
vercel env add NEXT_PUBLIC_API_URL production
# Enter: https://your-backend-url.railway.app

vercel env add NEXT_PUBLIC_WS_URL production
# Enter: wss://your-backend-url.railway.app

# Redeploy with new env vars
vercel --prod
```

### Update Backend CORS

```bash
cd ../backend
railway variables set CORS_ORIGINS="https://your-frontend.vercel.app,http://localhost:3001"
```

---

## Cost Estimates

### Free Tier Limits

**Railway**:
- $5 credit included
- ~500 hours of usage
- Enough for 1-2 months of light usage

**Vercel**:
- 100GB bandwidth/month
- Unlimited deployments
- Usually sufficient for moderate traffic

### Operational Costs

**With Rate Limiting (3 requests/hour)**:
- API costs: $20-50/month (100 users/week)
- Hosting: Free tier sufficient

**Heavy Usage (1000 users/week)**:
- API costs: $150-300/month
- Hosting: $10-20/month (Railway paid tier)

**Cost Optimization**:
- Use BYOK (Bring Your Own Key) for power users
- Enable caching for repeated questions
- Monitor usage with Railway/Vercel dashboards

---

## Monitoring & Maintenance

### Railway Dashboard

Monitor backend:
- **Metrics**: CPU, Memory, Network
- **Logs**: Real-time application logs
- **Usage**: API call volume
- Access at: https://railway.app/project/your-project-id

### Vercel Dashboard

Monitor frontend:
- **Analytics**: Page views, performance
- **Logs**: Function executions
- **Bandwidth**: Data transfer
- Access at: https://vercel.com/dashboard

### Set Up Alerts

**Railway**:
1. Go to project settings
2. Set up usage alerts (e.g., warn at $3 spent)
3. Add email notifications

**Vercel**:
1. Enable email notifications for deployments
2. Set up bandwidth alerts

---

## Troubleshooting

### Backend Issues

**Problem**: `500 Internal Server Error`
- Check Railway logs: `railway logs`
- Verify `ANTHROPIC_API_KEY` is set correctly
- Check if all dependencies installed

**Problem**: `CORS error` in browser
- Verify `CORS_ORIGINS` includes your frontend URL
- Check URL format (no trailing slash)
- Restart backend after changing env vars

**Problem**: `Rate limit exceeded`
- Increase `RATE_LIMIT_REQUESTS` in Railway variables
- Or wait for reset window (default 1 hour)

### Frontend Issues

**Problem**: `Failed to fetch` or network errors
- Verify `NEXT_PUBLIC_API_URL` is correct
- Check backend is running (visit backend URL directly)
- Verify WebSocket URL uses `wss://` not `ws://`

**Problem**: Build fails
- Check Vercel build logs
- Verify `package.json` and `package-lock.json` are committed
- Try rebuilding: `vercel --prod --force`

**Problem**: Environment variables not working
- Ensure variables start with `NEXT_PUBLIC_`
- Redeploy after adding env vars
- Clear Vercel cache: Settings → Advanced → Clear Cache

### General Issues

**Problem**: Slow responses
- Check API rate limits (Anthropic)
- Monitor Railway resource usage
- Consider upgrading to Haiku → Sonnet for slower but cheaper responses

**Problem**: High costs
- Review Railway usage metrics
- Lower rate limits
- Enable caching
- Consider BYOK mode

---

## Security Best Practices

1. **Never commit API keys**
   - Keep `.env` in `.gitignore` (already done)
   - Use environment variables only

2. **Monitor usage**
   - Set up cost alerts
   - Review logs weekly
   - Watch for abuse patterns

3. **Rate limiting**
   - Keep default 3 requests/hour for free tier
   - Increase only for trusted users
   - Use BYOK for unlimited access

4. **CORS configuration**
   - Only include your actual domains
   - Remove localhost in production
   - Update when changing frontend URL

5. **Update dependencies**
   - Run `npm audit fix` for frontend
   - Keep Python packages updated
   - Monitor security advisories

---

## Updating Deployed App

### To Update Backend

**Via Railway Dashboard**:
1. Push code to GitHub
2. Railway auto-deploys (if connected to main branch)

**Via CLI**:
```bash
cd web-tools/multi-agent-arena/backend
git push
railway up
```

### To Update Frontend

**Via Vercel Dashboard**:
1. Push code to GitHub
2. Vercel auto-deploys

**Via CLI**:
```bash
cd web-tools/multi-agent-arena/frontend
git push
vercel --prod
```

### Rolling Back

**Railway**:
1. Go to Deployments tab
2. Click "..." on previous deployment
3. Select "Redeploy"

**Vercel**:
1. Go to Deployments tab
2. Click "..." on previous deployment
3. Select "Promote to Production"

---

## Next Steps

After successful deployment:

1. **Test thoroughly**
   - Try all 4 strategies
   - Test on mobile devices
   - Verify rate limiting works

2. **Share with collaborators**
   - Send them the Vercel URL
   - Provide usage instructions
   - Set expectations on rate limits

3. **Monitor for first week**
   - Check costs daily
   - Review error logs
   - Gather user feedback

4. **Consider enhancements**
   - Add caching for popular questions
   - Implement user accounts
   - Add more agent personas
   - Build SELPHI Playground (Theory of Mind)

---

## Quick Reference

### URLs
- **Railway Dashboard**: https://railway.app/dashboard
- **Vercel Dashboard**: https://vercel.com/dashboard
- **Anthropic Console**: https://console.anthropic.com/

### Default Environment Variables

**Backend (Railway)**:
```bash
ANTHROPIC_API_KEY=sk-ant-...
CORS_ORIGINS=https://your-frontend.vercel.app
RATE_LIMIT_REQUESTS=3
RATE_LIMIT_WINDOW=3600
MAX_AGENTS=5
MAX_ROUNDS=3
DEFAULT_MODEL=claude-3-haiku-20240307
ALLOW_MODEL_SELECTION=false
ENV=production
LOG_LEVEL=INFO
```

**Frontend (Vercel)**:
```bash
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
NEXT_PUBLIC_WS_URL=wss://your-backend.railway.app
```

### Useful Commands

```bash
# Railway
railway login
railway logs
railway variables
railway status

# Vercel
vercel login
vercel logs
vercel env ls
vercel ls

# Local testing
cd backend && uvicorn app.main:app --reload
cd frontend && npm run dev
```

---

## Support

- **Railway Docs**: https://docs.railway.app/
- **Vercel Docs**: https://vercel.com/docs
- **Multi-Agent Arena Issues**: https://github.com/lfhvn/hidden-layer/issues

---

**Ready to deploy?** Start with Option 1 (Web Dashboards) for your first deployment!
