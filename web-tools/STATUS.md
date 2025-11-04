# Web Tools - Implementation Status

## 🎉 Completed

### Infrastructure (✅ Done)

- **Shared Backend Utilities**
  - Rate limiting system (IP-based)
  - API key validation (BYOK mode)
  - CORS middleware
  - Error handling
  - Logging setup
  - Cache utilities (stub)

- **Documentation**
  - Main README with philosophy and structure
  - Deployment guide with cost analysis
  - Quick start guide
  - Shared utilities documentation

### Multi-Agent Arena (✅ MVP Complete)

**Backend:**
- ✅ FastAPI app with WebSocket support
- ✅ Integration with research code (`projects/multi-agent/`)
- ✅ Rate limiting (3 requests/hour default)
- ✅ Strategy endpoints (debate, consensus, CRIT, manager-worker)
- ✅ Usage tracking endpoint
- ✅ Error handling
- ✅ BYOK mode support

**Frontend:**
- ✅ Next.js 14 app with TypeScript
- ✅ Strategy selector component
- ✅ Question input with examples
- ✅ Real-time usage indicator
- ✅ Result viewer with copy/share
- ✅ Responsive design (Tailwind CSS)
- ✅ Loading states and error handling

**DevOps:**
- ✅ Docker Compose setup
- ✅ Dockerfiles (frontend + backend)
- ✅ Makefile with common commands
- ✅ Environment configuration
- ✅ Development workflow

**Status**: Ready for local testing!

---

## 🚧 In Progress

### Steerability Dashboard

**Current State:**
- Existing frontend/backend at `projects/steerability/`
- Needs migration to `web-tools/` structure
- Needs polish for public deployment

**TODO:**
- [ ] Move to `web-tools/steerability/`
- [ ] Integrate shared rate limiting
- [ ] Add usage indicator
- [ ] Polish UI
- [ ] Add deployment configs

**Estimated Time**: 2-3 hours

---

## 📋 Planned

### Latent Lens Explorer

**Approach**: Read-only explorer of pre-trained SAE features
- Gallery view of discovered features
- Text analyzer (which features activate)
- Model comparison

**Status**: Research code exists at `projects/latent-space/lens/`

**TODO:**
- [ ] Pre-train SAE features
- [ ] Build read-only web interface
- [ ] Deploy static frontend + simple API

**Estimated Time**: 1 day

### SELPHI Playground

**Approach**: Interactive Theory of Mind scenarios
- Story-based UI for ToM tasks
- Multi-model comparison
- Leaderboard

**Status**: Backend exists at `projects/selphi/`

**TODO:**
- [ ] Build scenario UI
- [ ] Model comparison interface
- [ ] Result visualization

**Estimated Time**: 1 day

---

## 🎯 Next Steps (Recommended Order)

### Immediate (This Week)

1. **Test Multi-Agent Arena locally** (30 min)
   ```bash
   cd web-tools/multi-agent-arena
   make setup
   make dev
   ```

2. **Polish Steerability** (2-3 hours)
   - Migrate to web-tools structure
   - Add rate limiting
   - Polish UI

3. **Deploy Multi-Agent Arena** (1-2 hours)
   - Deploy backend to Railway
   - Deploy frontend to Vercel
   - Test in production

### Near Term (Next Week)

4. **Latent Lens Explorer** (1 day)
   - Pre-train features
   - Build read-only interface

5. **SELPHI Playground** (1 day)
   - Build scenario interface
   - Add model comparison

### Future

6. **Enhanced Features**
   - WebSocket streaming (real-time agent messages)
   - Agent personas
   - Tournament mode
   - Community voting
   - Transcript sharing

7. **Additional Tools**
   - Introspection playground (if differentiated from steerability)
   - AI-to-AI communication visualizer (when research progresses)

---

## 🔧 Known Issues / TODOs

### Multi-Agent Arena

- [ ] **Streaming**: Currently returns full result at end. Need to stream agent messages in real-time via WebSocket.
  - Requires modifying research code to yield intermediate results
  - Or wrapping API calls with streaming

- [ ] **Caching**: No caching yet. Should cache popular questions.
  - Add Redis or in-memory cache
  - Cache key: (question, strategy, n_agents)

- [ ] **Better error messages**: Generic errors shown to user
  - Add specific error types
  - Better UI for errors

- [ ] **Model selection**: Hardcoded to Haiku
  - Add model selector (Haiku free, Sonnet paid/BYOK)

- [ ] **Agent personas**: All agents use same system prompt
  - Add persona system (optimist, skeptic, pragmatist)

### Infrastructure

- [ ] **Frontend shared components**: Should extract to `web-tools/shared/frontend/`
  - UsageIndicator
  - ModelSelector
  - Common layout components

- [ ] **Better caching**: Add Redis support
- [ ] **Monitoring**: Add Sentry, PostHog
- [ ] **Testing**: Add unit/integration tests

---

## 📊 Cost Estimates

### Current (Free Tier + Rate Limits)

| Tool | Hosting | API Costs (100 users/week) |
|------|---------|---------------------------|
| Multi-Agent Arena | Free (Vercel + Railway credit) | $15-50 |
| Steerability | Free | $5-15 |

**Total**: ~$20-65/month

### Scaling (1000 users/week)

| Tool | Hosting | API Costs |
|------|---------|-----------|
| Multi-Agent Arena | $10-20 | $150-300 |
| Steerability | $10-20 | $50-100 |
| Latent Lens | Free (static) | $0-20 |
| SELPHI | Free | $20-50 |

**Total**: ~$240-510/month

With caching and BYOK adoption: ~$100-200/month

---

## 🎨 Design Philosophy

**Maintained**:
- ✅ Separate from research code
- ✅ Import from `projects/`, don't duplicate
- ✅ Shared infrastructure
- ✅ Clear rate limiting
- ✅ Easy deployment
- ✅ Doesn't interfere with research workflows

**Principles Working Well**:
- Research code stays pure
- Web layer is thin wrapper
- Can iterate independently
- Tools are genuinely useful (not just demos)

---

## 🤔 Open Questions

1. **Streaming Strategy**: Should we modify research code to stream, or wrap it?
   - **Option A**: Modify `projects/multi-agent/code/strategies.py` to yield
   - **Option B**: Wrap API calls in web backend to stream chunks
   - **Recommendation**: Option B (less invasive to research)

2. **Authentication**: Beyond rate limiting?
   - **Current**: IP-based rate limiting + BYOK
   - **Future**: User accounts? (Probably overkill for now)

3. **Monetization**: Free vs paid?
   - **Current**: Free with limits + BYOK unlimited
   - **Future**: Paid tier? ($5/mo unlimited?)

4. **Community Features**: Voting, sharing, leaderboards?
   - Would increase engagement
   - But adds complexity
   - **Recommendation**: Add after initial launch

---

## 📝 Feedback Welcome

Try it out and share thoughts on:
- UI/UX improvements
- Missing features
- Bugs
- Cost optimization ideas
- Additional tools to build

---

Last Updated: 2025-11-04
