# AgentMesh Web UI

Next.js frontend for AgentMesh.

## Setup

```bash
cd web

# Install dependencies
npm install

# Start dev server
npm run dev
```

Open http://localhost:3000

## Features

- **Home** - Overview, quick start, available strategies
- **Workflows** - List, create, view, delete workflows
- **Runs** - Execute workflows, view results with timeline
- **Real-time updates** - Run status polls every 2 seconds

## Stack

- Next.js 14 (App Router)
- TypeScript
- TailwindCSS
- Axios for API calls
- Lucide React for icons

## API Integration

The web UI talks to the FastAPI backend running on port 8000.

API client: `lib/api.ts`

## Development

```bash
# Install
npm install

# Dev server (with hot reload)
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Environment

Create `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## Routes

- `/` - Home
- `/workflows` - Workflow list
- `/workflows/new` - Create workflow
- `/workflows/[id]` - Workflow detail (TODO)
- `/workflows/[id]/run` - Execute workflow (TODO)
- `/runs/[id]` - Run detail with timeline

## TODO

- [ ] Workflow detail page with graph visualization
- [ ] Execute workflow form
- [ ] Better graph editor (drag & drop)
- [ ] Authentication
- [ ] Dark mode
- [ ] Error boundaries
- [ ] Loading states
- [ ] Pagination for large lists
