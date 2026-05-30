# Topologies Implementation Plan - Practical MVP

## Executive Summary

**Current State**: Excellent planning docs, working data pipeline scripts, **zero mobile app code**

**Recommendation**: Build a **web-based MVP first** before mobile. This lets you:
- Ship something interactive in 1-2 weeks instead of 12
- Validate the core experience without React Native complexity
- Reuse existing infrastructure (you already have web stack from Lens)
- Port to mobile once proven

**Timeline**:
- **Web MVP**: 1-2 weeks (recommended start)
- **Mobile MVP**: 4-6 weeks (after web validation)
- **Full Mobile App**: 12 weeks (original plan)

---

## Current State Assessment

### ✅ What Exists

**Documentation** (Comprehensive):
- `PRD.md` - Product requirements and 12-week timeline
- `TECH_PLAN.md` - Architecture and stack decisions
- `UX_STORYBOOK.md` - Mobile UX flows
- `SETUP_DEV.md` - Development environment guide
- `TASKS.md` - Milestone breakdown
- `AUDIO_HAPTIC_SPEC.md` - Audio/haptic specifications
- `MODEL_CONVERSION.md` - Core ML/TFLite conversion
- `DATA_CORPUS_GUIDE.md` - Corpus guidelines
- `INTEGRATION.md` - Integration plans

**Data Pipeline** (Working):
- `scripts/embed_corpus.py` - **Excellent quality**
  - Sentence transformers embedding
  - UMAP projection
  - Checkpointing
  - Batch processing
  - Logging and validation
- `scripts/build_seed_db.py` - Database builder
- `scripts/generate_corpus.py` - Corpus generator
- `notebooks/01_embedding_exploration.ipynb` - Exploration notebook
- `data/corpus.csv` - Sample corpus (249 items)

**Infrastructure**:
- Virtual environment setup
- requirements.txt with dependencies
- .gitignore configured

### ❌ What's Missing

**Mobile App** (Nothing implemented):
- ❌ No React Native project
- ❌ No Expo configuration
- ❌ No mobile UI components
- ❌ No 3D visualization
- ❌ No audio/haptic system
- ❌ No on-device model inference
- ❌ No Core ML/TFLite conversion

**Gap**: 100% of mobile implementation

---

## The Problem with the Current Plan

### Original Plan Issues

1. **High Complexity**: React Native + Three.js + Audio + Haptics + On-device ML
2. **Long Timeline**: 12 weeks to first usable version
3. **Platform Lock-in**: iOS/Android only (can't share easily)
4. **Testing Friction**: Requires physical device or simulator
5. **Distribution Barrier**: App store submission for wider testing

### Why This Matters

**You want to experience latent space** - but the path to that experience is very long.

---

## Recommended Approach: Web-First MVP

### Why Web First?

**Faster to Ship**:
- Reuse Lens web stack (Next.js + FastAPI)
- No React Native learning curve
- No mobile deployment complexity
- 1-2 weeks vs 12 weeks

**Easier to Share**:
- Send a link, anyone can try it
- No app store submission
- Works on any device with browser
- Can still work on mobile browsers

**Validate Core Experience**:
- Test if exploring latent space is actually compelling
- Iterate on visualization rapidly
- Try different corpus types
- Get user feedback quickly

**Port Later**:
- Once web version works, port to React Native
- Reuse backend completely
- Reuse design patterns
- Add mobile-specific features (haptics, offline)

### Web MVP Features

**Core Experience** (Week 1):
- ✅ 2D interactive map (Canvas or Three.js)
- ✅ Pan/zoom navigation
- ✅ Click point → see text and neighbors
- ✅ Search functionality
- ✅ Live backend (FastAPI serving embeddings/coords)

**Enhanced** (Week 2):
- ✅ Interpolation between two points
- ✅ Cluster visualization (color by density/topic)
- ✅ Creator mode (add your own text)
- ✅ Export map as image/data

**Mobile-Specific** (Port Later):
- ⏭ Audio mapping (can prototype with Web Audio API)
- ⏭ Haptic feedback (mobile only)
- ⏭ Offline mode (mobile only)
- ⏭ On-device embedding (mobile only)

---

## Architecture Comparison

### Web MVP Architecture

```
┌──────────────────────────┐
│   Next.js Frontend       │
│   - Canvas/Three.js map  │
│   - Search & filters     │
│   - Interpolation UI     │
└────────────┬─────────────┘
             │ HTTP/WebSocket
┌────────────▼─────────────┐
│   FastAPI Backend        │
│   - Serve embeddings     │
│   - Compute neighbors    │
│   - Embed new text       │
│   - UMAP projection      │
└────────────┬─────────────┘
             │
┌────────────▼─────────────┐
│   SQLite + NumPy Files   │
│   - corpus.csv           │
│   - embeddings.npy       │
│   - coords_umap.npy      │
│   - edges (kNN graph)    │
└──────────────────────────┘
```

**Stack**:
- Frontend: Next.js 14, TypeScript, Three.js or D3.js
- Backend: FastAPI, sentence-transformers, UMAP
- Data: SQLite (metadata) + NumPy files (vectors)
- Deployment: Docker Compose (like Lens)

### Original Mobile Architecture

```
┌──────────────────────────┐
│   React Native + Expo    │
│   - react-three-fiber    │
│   - expo-av (audio)      │
│   - expo-haptics         │
└────────────┬─────────────┘
             │
┌────────────▼─────────────┐
│   On-Device Inference    │
│   - Core ML (iOS)        │
│   - TFLite (Android)     │
│   - Quantized model      │
└────────────┬─────────────┘
             │
┌────────────▼─────────────┐
│   SQLite (on device)     │
│   - Pre-bundled corpus   │
│   - Embeddings + coords  │
│   - kNN graph            │
└──────────────────────────┘
```

**Challenges**:
- Core ML conversion non-trivial
- React Native + Three.js integration tricky
- Offline-first adds complexity
- Platform-specific code (iOS vs Android)

---

## Implementation Plan - Web MVP

### Phase 0: Setup (Day 1)

**Goal**: Project structure and data ready

**Tasks**:

1. **Create Web App Directory**
```bash
cd /home/user/hidden-layer/representations/latent-space/topologies
mkdir -p web-app/{backend,frontend}
```

2. **Run Data Pipeline**
```bash
# Activate Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate embeddings and UMAP projection
python scripts/embed_corpus.py \
  --model all-MiniLM-L6-v2 \
  --input data/corpus.csv \
  --embeddings data/embeddings.npy \
  --coords data/coords_umap.npy \
  --umap \
  --save-umap-model data/umap_model.pkl

# Build SQLite database
python scripts/build_seed_db.py \
  --corpus data/corpus.csv \
  --embeddings data/embeddings.npy \
  --coords data/coords_umap.npy \
  --out data/seed.db \
  --knn 12
```

3. **Validate Data**
```bash
# Check output files
ls -lh data/*.npy data/*.db

# Inspect database
sqlite3 data/seed.db << EOF
.schema
SELECT COUNT(*) FROM item;
SELECT COUNT(*) FROM coord2d;
SELECT COUNT(*) FROM edge;
.quit
EOF
```

**Deliverable**: Working embeddings, UMAP coords, SQLite DB

---

### Phase 1: Backend API (Days 2-3)

**Goal**: FastAPI server exposing latent space data

**Create** `web-app/backend/app/main.py`:

```python
"""
Topologies Backend API
Serves latent space data for visualization
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import sqlite3
from pathlib import Path
import pickle

app = FastAPI(title="Topologies API", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data on startup
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH = DATA_DIR / "seed.db"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
COORDS_PATH = DATA_DIR / "coords_umap.npy"
UMAP_MODEL_PATH = DATA_DIR / "umap_model.pkl"

# Global state
embeddings: Optional[np.ndarray] = None
coords: Optional[np.ndarray] = None
umap_model = None

@app.on_event("startup")
async def load_data():
    """Load embeddings and coordinates into memory"""
    global embeddings, coords, umap_model

    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"✓ Loaded embeddings: {embeddings.shape}")

    print("Loading coordinates...")
    coords = np.load(COORDS_PATH)
    print(f"✓ Loaded coordinates: {coords.shape}")

    if UMAP_MODEL_PATH.exists():
        print("Loading UMAP model...")
        with open(UMAP_MODEL_PATH, 'rb') as f:
            umap_model = pickle.load(f)
        print("✓ Loaded UMAP model")

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# === API Routes ===

@app.get("/")
async def root():
    return {
        "message": "Topologies API",
        "version": "0.1.0",
        "corpus_size": len(coords) if coords is not None else 0
    }

@app.get("/points")
async def get_all_points():
    """Get all points with coordinates and metadata"""
    conn = get_db()
    cursor = conn.execute("""
        SELECT i.id, i.text, i.meta, c.x, c.y
        FROM item i
        JOIN coord2d c ON i.id = c.item_id
    """)

    points = []
    for row in cursor:
        points.append({
            "id": row["id"],
            "text": row["text"],
            "meta": row["meta"],
            "x": row["x"],
            "y": row["y"]
        })

    conn.close()
    return {"points": points, "count": len(points)}

@app.get("/point/{item_id}")
async def get_point_detail(item_id: int):
    """Get detailed info for a single point including neighbors"""
    conn = get_db()

    # Get item info
    cursor = conn.execute(
        "SELECT id, text, meta FROM item WHERE id = ?",
        (item_id,)
    )
    item = cursor.fetchone()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    # Get coordinates
    cursor = conn.execute(
        "SELECT x, y FROM coord2d WHERE item_id = ?",
        (item_id,)
    )
    coord = cursor.fetchone()

    # Get neighbors
    cursor = conn.execute("""
        SELECT i.id, i.text, e.weight
        FROM edge e
        JOIN item i ON e.dst = i.id
        WHERE e.src = ?
        ORDER BY e.weight DESC
        LIMIT 10
    """, (item_id,))

    neighbors = []
    for row in cursor:
        neighbors.append({
            "id": row["id"],
            "text": row["text"],
            "similarity": row["weight"]
        })

    conn.close()

    return {
        "id": item["id"],
        "text": item["text"],
        "meta": item["meta"],
        "x": coord["x"] if coord else None,
        "y": coord["y"] if coord else None,
        "neighbors": neighbors
    }

@app.get("/search")
async def search(q: str, limit: int = 20):
    """Text search in corpus"""
    conn = get_db()
    cursor = conn.execute("""
        SELECT i.id, i.text, c.x, c.y
        FROM item i
        JOIN coord2d c ON i.id = c.item_id
        WHERE i.text LIKE ?
        LIMIT ?
    """, (f"%{q}%", limit))

    results = []
    for row in cursor:
        results.append({
            "id": row["id"],
            "text": row["text"],
            "x": row["x"],
            "y": row["y"]
        })

    conn.close()
    return {"results": results, "count": len(results)}

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed_text(request: EmbedRequest):
    """Embed new text and place on map"""
    from sentence_transformers import SentenceTransformer

    # Load model (cache this in production)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed
    embedding = model.encode([request.text])[0]

    # Project to 2D using UMAP model
    if umap_model:
        coord_2d = umap_model.transform([embedding])[0]
    else:
        # Fallback: find nearest neighbor
        similarities = embeddings @ embedding
        nearest_idx = np.argmax(similarities)
        coord_2d = coords[nearest_idx] + np.random.randn(2) * 0.01

    return {
        "text": request.text,
        "x": float(coord_2d[0]),
        "y": float(coord_2d[1]),
        "embedding_dim": len(embedding)
    }

class InterpolateRequest(BaseModel):
    id1: int
    id2: int
    alpha: float  # 0.0 to 1.0

@app.post("/interpolate")
async def interpolate_points(request: InterpolateRequest):
    """Interpolate between two points in latent space"""
    if not (0 <= request.alpha <= 1):
        raise HTTPException(status_code=400, detail="Alpha must be between 0 and 1")

    # Get embeddings for both points
    emb1 = embeddings[request.id1]
    emb2 = embeddings[request.id2]

    # Linear interpolation in embedding space
    interp_embedding = (1 - request.alpha) * emb1 + request.alpha * emb2

    # Project to 2D
    if umap_model:
        coord_2d = umap_model.transform([interp_embedding])[0]
    else:
        # Linear interpolation in 2D space
        coord_2d = (1 - request.alpha) * coords[request.id1] + request.alpha * coords[request.id2]

    return {
        "x": float(coord_2d[0]),
        "y": float(coord_2d[1]),
        "alpha": request.alpha
    }
```

**Create** `web-app/backend/requirements.txt`:
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
sentence-transformers==2.2.2
numpy==1.24.3
pydantic==2.5.0
umap-learn==0.5.5
```

**Test Backend**:
```bash
cd web-app/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001

# Test endpoints
curl http://localhost:8001/
curl http://localhost:8001/points | jq '.count'
curl http://localhost:8001/point/1 | jq '.neighbors'
```

**Deliverable**: Working API serving latent space data

---

### Phase 2: Frontend Visualization (Days 4-6)

**Goal**: Interactive 2D map of latent space

**Create Next.js App**:
```bash
cd web-app
npx create-next-app@latest frontend --typescript --tailwind --app
cd frontend
npm install three @react-three/fiber @react-three/drei
npm install zustand  # State management
```

**Key Components**:

1. **`src/components/LatentMap.tsx`** - Main 2D visualization
2. **`src/components/PointDetail.tsx`** - Detail drawer
3. **`src/components/SearchBar.tsx`** - Search interface
4. **`src/components/InterpolationControls.tsx`** - Interpolation UI

**Visualization Options**:

**Option A: Canvas 2D** (Simpler, faster to implement)
- Use HTML5 Canvas
- Point-and-click scatter plot
- Pan/zoom with mouse
- Fastest rendering for 10k+ points

**Option B: Three.js** (More beautiful, closer to mobile vision)
- 2D points in 3D space (can add 3D later)
- Better animations
- More impressive visually
- Slightly more complex

**Recommendation**: Start with Canvas 2D, upgrade to Three.js in week 2

**Example Canvas Implementation** (`src/components/LatentMap.tsx`):

```typescript
'use client';

import { useEffect, useRef, useState } from 'react';

interface Point {
  id: number;
  text: string;
  x: number;
  y: number;
}

interface LatentMapProps {
  points: Point[];
  onPointClick: (point: Point) => void;
  highlightedId?: number;
}

export default function LatentMap({ points, onPointClick, highlightedId }: LatentMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [hoveredPoint, setHoveredPoint] = useState<Point | null>(null);

  // Render loop
  useEffect(() => {
    if (!canvasRef.current || points.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    // Clear
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Find bounds
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    // Transform to screen coordinates
    const toScreen = (x: number, y: number) => {
      const normX = (x - minX) / (maxX - minX);
      const normY = (y - minY) / (maxY - minY);

      const padding = 50;
      const screenX = padding + normX * (canvas.width - 2 * padding);
      const screenY = padding + normY * (canvas.height - 2 * padding);

      // Apply pan/zoom
      return {
        x: (screenX - canvas.width / 2) * transform.scale + canvas.width / 2 + transform.x,
        y: (screenY - canvas.height / 2) * transform.scale + canvas.height / 2 + transform.y
      };
    };

    // Draw points
    points.forEach(point => {
      const screen = toScreen(point.x, point.y);

      // Point styling
      const isHighlighted = point.id === highlightedId;
      const isHovered = hoveredPoint?.id === point.id;

      ctx.beginPath();
      ctx.arc(screen.x, screen.y, isHighlighted ? 8 : isHovered ? 6 : 3, 0, Math.PI * 2);

      if (isHighlighted) {
        ctx.fillStyle = '#60a5fa';  // Blue
        ctx.strokeStyle = '#93c5fd';
        ctx.lineWidth = 2;
        ctx.stroke();
      } else if (isHovered) {
        ctx.fillStyle = '#fbbf24';  // Yellow
      } else {
        ctx.fillStyle = '#4b5563';  // Gray
      }

      ctx.fill();
    });

    // Draw hovered point label
    if (hoveredPoint) {
      const screen = toScreen(hoveredPoint.x, hoveredPoint.y);
      ctx.fillStyle = '#fff';
      ctx.font = '12px monospace';
      ctx.fillText(
        hoveredPoint.text.substring(0, 50) + '...',
        screen.x + 10,
        screen.y - 10
      );
    }

  }, [points, transform, highlightedId, hoveredPoint]);

  // Mouse handlers
  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Find clicked point (simplified - should use proper coordinate transform)
    // TODO: Implement proper screen-to-world coordinate transformation

    // For now, find closest point
    let closest: Point | null = null;
    let closestDist = Infinity;

    points.forEach(point => {
      // This is a placeholder - need proper coordinate transform
      const dist = Math.hypot(mouseX - point.x * 10, mouseY - point.y * 10);
      if (dist < closestDist && dist < 20) {
        closestDist = dist;
        closest = point;
      }
    });

    if (closest) {
      onPointClick(closest);
    }
  };

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-pointer"
        onClick={handleClick}
        onMouseMove={(e) => {
          // TODO: Implement hover detection
        }}
      />

      {/* Zoom controls */}
      <div className="absolute bottom-4 right-4 flex flex-col gap-2">
        <button
          onClick={() => setTransform(t => ({ ...t, scale: t.scale * 1.2 }))}
          className="bg-gray-800 text-white px-3 py-2 rounded"
        >
          +
        </button>
        <button
          onClick={() => setTransform(t => ({ ...t, scale: t.scale / 1.2 }))}
          className="bg-gray-800 text-white px-3 py-2 rounded"
        >
          −
        </button>
        <button
          onClick={() => setTransform({ x: 0, y: 0, scale: 1 })}
          className="bg-gray-800 text-white px-2 py-1 rounded text-sm"
        >
          Reset
        </button>
      </div>
    </div>
  );
}
```

**Main Page** (`src/app/page.tsx`):

```typescript
'use client';

import { useState, useEffect } from 'react';
import LatentMap from '@/components/LatentMap';
import PointDetail from '@/components/PointDetail';
import SearchBar from '@/components/SearchBar';

const API_BASE = 'http://localhost:8001';

export default function Home() {
  const [points, setPoints] = useState([]);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [loading, setLoading] = useState(true);

  // Load all points on mount
  useEffect(() => {
    fetch(`${API_BASE}/points`)
      .then(res => res.json())
      .then(data => {
        setPoints(data.points);
        setLoading(false);
      })
      .catch(err => console.error('Failed to load points:', err));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-2xl">Loading latent space...</div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-black text-white">
      {/* Left: Map */}
      <div className="flex-1 relative">
        <div className="absolute top-4 left-4 z-10">
          <h1 className="text-3xl font-bold mb-2">Latent Topologies</h1>
          <p className="text-gray-400">{points.length} points</p>
        </div>

        <SearchBar onSearch={(query) => {
          // TODO: Implement search
        }} />

        <LatentMap
          points={points}
          onPointClick={setSelectedPoint}
          highlightedId={selectedPoint?.id}
        />
      </div>

      {/* Right: Detail panel */}
      {selectedPoint && (
        <PointDetail
          point={selectedPoint}
          onClose={() => setSelectedPoint(null)}
        />
      )}
    </div>
  );
}
```

**Deliverable**: Interactive web map of latent space

---

### Phase 3: Creator Mode & Interpolation (Day 7)

**Goal**: Add your own text and explore interpolations

**Features**:
1. Text input → embed → place on map
2. Select two points → interpolation slider
3. See blended position in real-time

**Implementation**: Add UI components that call `/embed` and `/interpolate` endpoints

**Deliverable**: Full interactive web experience

---

## Next Steps After Web MVP

Once web MVP works, you have options:

### Option A: Port to React Native (4-6 weeks)

Follow original plan:
- Convert Next.js components to React Native
- Add react-three-fiber for 3D
- Implement audio (expo-av)
- Implement haptics (expo-haptics)
- Convert model to Core ML
- Add offline support

### Option B: Enhance Web Version (2-4 weeks)

Add advanced features:
- 3D visualization (Three.js)
- Web Audio API for sound mapping
- Temporal tracking (save exploration paths)
- Collaborative mode (multiple users)
- Different corpora (wiki, quotes, code, etc.)

### Option C: Hybrid Approach

- Keep web version as primary
- Make it mobile-responsive
- Add PWA support (installable)
- Later: Create simplified React Native version

---

## Resource Requirements

### Web MVP

**Skills Needed**:
- Python (FastAPI) - already have
- TypeScript/Next.js - already have (from Lens)
- Canvas or Three.js - new, but good tutorials
- Data viz principles - learnable

**Time**:
- Phase 0: 1 day (data pipeline)
- Phase 1: 2 days (backend API)
- Phase 2: 3 days (frontend map)
- Phase 3: 1 day (creator mode)
- **Total: ~7 days of focused work**

**Infrastructure**:
- Reuse Lens deployment setup
- Docker Compose
- Same hosting strategy

### Mobile MVP (If Pursued)

**Skills Needed**:
- React Native + Expo - new
- react-three-fiber - new
- Core ML Tools - new
- Mobile dev patterns - new

**Time**:
- M0: 2 weeks (model conversion)
- M1: 2 weeks (map viz)
- M2: 2 weeks (audio/haptics)
- M3: 3 weeks (creator mode)
- M4: 2 weeks (annotations)
- M5: 1 week (polish)
- **Total: 12 weeks**

---

## Recommendation

**Start with Web MVP**:

1. **This Week**: Build backend API (Phase 1)
2. **Next Week**: Build frontend map (Phase 2 + 3)
3. **Week 3**: Get user feedback, iterate
4. **Week 4**: Decide - enhance web or port to mobile?

**Why**:
- Validate the core experience quickly
- Ship something shareable immediately
- Learn what works before committing to mobile complexity
- Reuse all your existing web infrastructure

**Mobile is great for**:
- Haptics (critical feature?)
- Offline (nice-to-have?)
- Portability (web works on mobile browsers)

**But mobile adds**:
- 8+ weeks of development time
- Platform-specific complexity
- Distribution friction
- Harder testing/iteration

Get the experience working on web first. Then decide if mobile-specific features justify the investment.

---

## Decision Point

**Question for you**:

Do you want to:

**A) Build web MVP first** (1-2 weeks, easier to share, reuses existing stack)

**B) Go straight to mobile** (12 weeks, more ambitious, matches original vision)

**C) Something else?**

I recommend A, but I'm ready to help with whichever you choose.

---

## Files Created

After this planning phase, I can create:

1. **Web MVP scaffold** (`web-app/` directory structure)
2. **Backend API** (complete FastAPI implementation)
3. **Frontend starter** (Next.js with basic components)
4. **Docker setup** (deployment configuration)
5. **Quick start guide** (step-by-step setup)

Or if you prefer mobile:

1. **React Native scaffold** (Expo project structure)
2. **Model conversion scripts** (Core ML/TFLite)
3. **Mobile component templates**
4. **Setup guide for mobile dev**

**What would you like to do?**
