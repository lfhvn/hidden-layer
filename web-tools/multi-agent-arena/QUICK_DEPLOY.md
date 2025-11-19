# Quick Deploy Checklist

15-minute deployment guide for Multi-Agent Arena.

## ✅ Pre-Deployment Checklist

- [ ] Anthropic API key ready (get at https://console.anthropic.com/)
- [ ] Railway account created (https://railway.app/)
- [ ] Vercel account created (https://vercel.com/)
- [ ] Code committed and pushed to GitHub ✓ (already done!)

---

## 🚀 Deploy Backend (5 minutes)

### 1. Create Railway Project

1. Go to **https://railway.app/dashboard**
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select: `lfhvn/hidden-layer`
4. **Root Directory**: `web-tools/multi-agent-arena/backend`

### 2. Set Environment Variables

Click "Variables" and paste this (replace API key):

```
ANTHROPIC_API_KEY=sk-ant-YOUR-KEY-HERE
CORS_ORIGINS=http://localhost:3001
RATE_LIMIT_REQUESTS=3
RATE_LIMIT_WINDOW=3600
ENV=production
```

### 3. Deploy & Get URL

1. Click **"Deploy"**
2. Wait ~2 minutes
3. Copy the Railway URL (looks like: `https://xxx.up.railway.app`)

### 4. Test It

```bash
curl https://YOUR-RAILWAY-URL.railway.app/
# Should see: {"status":"ok"...}
```

✅ **Backend URL**: ___________________ (save this!)

---

## 🌐 Deploy Frontend (5 minutes)

### 1. Create Vercel Project

1. Go to **https://vercel.com/dashboard**
2. Click **"Add New..."** → **"Project"**
3. Import: `lfhvn/hidden-layer`
4. **Root Directory**: `web-tools/multi-agent-arena/frontend`
5. **Framework**: Next.js (auto-detected)

### 2. Set Environment Variables

Before deploying, add these (use YOUR Railway URL from above):

```
NEXT_PUBLIC_API_URL=https://YOUR-RAILWAY-URL.railway.app
NEXT_PUBLIC_WS_URL=wss://YOUR-RAILWAY-URL.railway.app
```

### 3. Deploy & Get URL

1. Click **"Deploy"**
2. Wait ~2 minutes
3. Copy the Vercel URL (looks like: `https://xxx.vercel.app`)

✅ **Frontend URL**: ___________________ (save this!)

---

## 🔗 Link Frontend & Backend (2 minutes)

### Update CORS on Backend

1. Go back to **Railway dashboard**
2. Your backend project → **"Variables"**
3. Update `CORS_ORIGINS` to:
   ```
   CORS_ORIGINS=https://YOUR-VERCEL-URL.vercel.app,http://localhost:3001
   ```
4. Save (auto-redeploys in ~1 minute)

---

## ✨ Test It! (3 minutes)

1. Visit your Vercel URL
2. Select strategy: **"Debate"**
3. Enter question: **"Should AI be regulated?"**
4. Click **"Start Debate"**

### Expected Result:
- ✅ You see agents debating in real-time
- ✅ Judge synthesizes final answer
- ✅ No errors in browser console (F12)

### If Something's Wrong:

**CORS Error?**
- Check Railway CORS_ORIGINS includes your Vercel URL
- Wait 1 minute for Railway to redeploy

**Can't connect to backend?**
- Verify `NEXT_PUBLIC_API_URL` is correct in Vercel
- Check Railway logs for errors

**500 Error?**
- Verify `ANTHROPIC_API_KEY` is set in Railway
- Check Railway logs

---

## 📱 Access from Mobile

Your Vercel URL works on any device:
- iPhone/iPad Safari
- Android Chrome
- Any mobile browser

Just visit: `https://YOUR-VERCEL-URL.vercel.app`

---

## 💰 Cost Tracking

**First Month (Free Tier)**:
- Railway: $5 credit (sufficient for testing)
- Vercel: Free unlimited
- API costs: ~$10-20 with rate limiting

**Monitor**:
- Railway dashboard: https://railway.app/dashboard
- Check API usage weekly

---

## 🎉 You're Done!

**Share your deployment**:
- URL: `https://YOUR-VERCEL-URL.vercel.app`
- Rate limit: 3 debates/hour (per IP)
- Strategies: Debate, Consensus, CRIT, Manager-Worker

**Next Steps**:
- [ ] Test on mobile device
- [ ] Share with collaborators
- [ ] Monitor costs for first week
- [ ] Build SELPHI Playground next!

---

## 📞 Need Help?

See full guide: `DEPLOYMENT_GUIDE.md`

**Common Issues**:
- Railway logs: `railway logs` (if CLI installed)
- Vercel logs: Project → Deployments → Click deployment → View Function Logs
- Railway dashboard: https://railway.app/project/YOUR-PROJECT
- Vercel dashboard: https://vercel.com/dashboard
