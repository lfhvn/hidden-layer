# Web Tools - Public Research Demos

Public-facing web interfaces for Hidden Layer research projects.

## Philosophy

**Research First, Demos Second**: These tools are public-facing interfaces that showcase research from `/projects/`. They:
- Import and use research code without modifying it
- Add user-friendly UIs, authentication, and rate limiting
- Can be deployed independently
- Don't interfere with notebook-based research workflows

## Available Tools

### 🎛️ Steerability Dashboard
**Status**: In Progress
**Path**: `steerability/`
**Description**: Interactive LLM steering with real-time adherence metrics

## Architecture

```
web-tools/
├── shared/                    # Shared infrastructure
│   ├── auth/                  # Rate limiting, API key management
│   ├── ui-components/         # Reusable React components
│   └── deployment/            # Docker, deployment configs
├── steerability/              # Individual tools
└── [tool-name]/
```

### Shared Infrastructure

All tools use:
- **Authentication**: Simple rate limiting + optional API keys
- **UI Components**: Consistent design system
- **Cost Controls**: Usage limits, caching
- **Deployment**: One-command deploy to cloud platforms

### Tool Structure

Each tool has:
```
tool-name/
├── frontend/              # Next.js/React
├── backend/               # FastAPI
├── docker-compose.yml     # Local development
├── Dockerfile             # Production builds
└── README.md              # Tool-specific docs
```

## Development Workflow

### Working on Research (Main Workflow)
```bash
cd communication/multi-agent/
jupyter notebook  # Do research as usual
```

### Building Public Interface
```bash
cd web-tools/steerability/
make dev  # Starts local dev environment
```

**Key Principle**: Web tools import from research projects, never modify them.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+
- Python 3.11+

### Run a Tool Locally
```bash
cd web-tools/steerability/
cp .env.example .env
make dev

# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

### Deploy to Production
```bash
cd web-tools/steerability/
make deploy  # Handles Vercel + Railway deployment
```

## Cost Management

Public tools implement:
1. **Rate Limiting**: 5 requests per IP per hour (free tier)
2. **API Key Mode**: Users can provide their own keys
3. **Caching**: Popular queries cached
4. **Model Selection**: Smaller models available for free tier

See `DEPLOYMENT.md` for cost analysis.

## Contributing

### Adding a New Tool

1. Create directory: `web-tools/my-tool/`
2. Copy template: `cp -r shared/template/ my-tool/`
3. Import research code: `from communication.multi_agent import ...` (or the relevant area package)
4. Build UI
5. Test locally: `make dev`
6. Deploy: `make deploy`

### Modifying Existing Tool

- **UI changes**: Edit `frontend/src/`
- **Add features**: Import from `projects/`, don't duplicate code
- **Cost optimization**: Update rate limits in `.env`

## Deployment

Each tool can be deployed independently:

- **Frontend**: Vercel (recommended) or Netlify
- **Backend**: Railway, Fly.io, or Cloud Run
- **Database**: Supabase (if needed)

See `DEPLOYMENT.md` for detailed instructions.

## Monitoring

- **Usage Tracking**: PostHog or Plausible
- **Error Tracking**: Sentry
- **Cost Alerts**: Cloud platform billing alerts

## Questions?

- **Bug reports**: GitHub Issues
- **Deployment help**: See `DEPLOYMENT.md`
- **New tool ideas**: Discuss in Issues

---

**Remember**: These are public demos of research. The real work happens in the research area directories (`communication/`, `alignment/`, `representations/`, ...).
