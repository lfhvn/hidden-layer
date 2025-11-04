# Shared Infrastructure

Reusable components for all web tools.

## Structure

```
shared/
├── backend/              # Python/FastAPI utilities
│   ├── auth.py          # Rate limiting, API keys
│   ├── middleware.py    # CORS, error handling
│   └── cache.py         # Response caching
├── frontend/            # React/TypeScript components
│   ├── components/      # UI components
│   ├── hooks/           # Custom hooks
│   └── utils/           # Helper functions
└── deployment/          # Docker configs
    ├── Dockerfile.backend
    ├── Dockerfile.frontend
    └── docker-compose.template.yml
```

## Backend Utilities

### Rate Limiting

```python
from web_tools.shared.backend.auth import RateLimiter, require_api_key

limiter = RateLimiter(requests=5, window=3600)

@app.post("/api/generate")
@limiter.limit("5/hour")
async def generate(request: Request):
    ...
```

### Caching

```python
from web_tools.shared.backend.cache import cache_response

@app.get("/api/models")
@cache_response(ttl=3600)
async def list_models():
    ...
```

## Frontend Components

### Usage Limiter

```tsx
import { UsageLimiter } from '@/web-tools/shared/frontend/components'

<UsageLimiter
  currentUsage={3}
  maxUsage={5}
  resetTime="1 hour"
/>
```

### Model Selector

```tsx
import { ModelSelector } from '@/web-tools/shared/frontend/components'

<ModelSelector
  models={['claude-3-haiku', 'claude-3-sonnet']}
  onChange={setModel}
/>
```

## Installation

### In a Tool's Backend

```python
# In tool's requirements.txt
../../shared/backend

# In tool's code
from web_tools.shared.backend import RateLimiter
```

### In a Tool's Frontend

```json
// In tool's package.json
{
  "dependencies": {
    "web-tools-shared": "file:../../shared/frontend"
  }
}
```

## Development

```bash
# Backend
cd shared/backend
pip install -e .
pytest

# Frontend
cd shared/frontend
npm install
npm run build
npm test
```
