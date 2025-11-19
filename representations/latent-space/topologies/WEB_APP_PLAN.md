# Web App Implementation Plan - Latent Topologies MVP

**Target:** Interactive 3D visualization of 2000 concept embeddings in browser

**Tech Stack:** Vite + React + TypeScript + Three.js (react-three-fiber)

**Timeline:** 2-3 days for core MVP

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (Client)                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐    ┌──────────────────┐            │
│  │  React UI      │    │  Three.js Scene  │            │
│  │  Components    │◄───┤  (Point Cloud)   │            │
│  └────────────────┘    └──────────────────┘            │
│         ▲                       ▲                       │
│         │                       │                       │
│         └───────────┬───────────┘                       │
│                     │                                   │
│              ┌──────▼────────┐                          │
│              │  Data Layer   │                          │
│              │  (Fetches DB) │                          │
│              └───────────────┘                          │
│                     ▲                                   │
└─────────────────────┼───────────────────────────────────┘
                      │
              ┌───────▼────────┐
              │  Static Files  │
              │  seed.db or    │
              │  coords.json   │
              └────────────────┘
```

---

## Phase 1: Project Setup (30 min)

### M1.1: Initialize Vite + React + TypeScript

```bash
cd representations/latent-space/topologies
npm create vite@latest web-app -- --template react-ts
cd web-app
npm install
```

**Expected structure:**
```
web-app/
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   └── vite-env.d.ts
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

### M1.2: Install Dependencies

```bash
npm install @react-three/fiber @react-three/drei three
npm install -D @types/three
```

**Dependencies:**
- `@react-three/fiber` - React renderer for Three.js
- `@react-three/drei` - Helper components (OrbitControls, etc.)
- `three` - 3D graphics library

**Test it works:**
```bash
npm run dev
# Open http://localhost:5173
```

---

## Phase 2: Data Loading (1-2 hours)

### M1.3: Export Database to JSON

We need to get data from SQLite into the browser. Two options:

**Option A: Export to JSON (Simpler for MVP)**

Create `scripts/export_for_web.py`:

```python
#!/usr/bin/env python3
"""Export SQLite database to JSON for web app."""
import sqlite3
import json
import sys

def export_to_json(db_path, output_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Get all items with coordinates
    cur.execute("""
        SELECT
            i.id,
            i.text,
            i.topic,
            c.x,
            c.y
        FROM item i
        JOIN coord2d c ON i.id = c.item_id
    """)

    items = []
    for row in cur.fetchall():
        items.append({
            "id": row[0],
            "text": row[1],
            "topic": row[2],
            "x": row[3],
            "y": row[4]
        })

    # Get kNN edges for each item
    edges = {}
    cur.execute("""
        SELECT src, dst, weight
        FROM edge
        ORDER BY src, weight DESC
    """)

    for src, dst, weight in cur.fetchall():
        if src not in edges:
            edges[src] = []
        edges[src].append({"id": dst, "similarity": weight})

    output = {
        "items": items,
        "edges": edges,
        "metadata": {
            "num_items": len(items),
            "num_edges": len(edges)
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output, f)

    print(f"✓ Exported {len(items)} items to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")

if __name__ == "__main__":
    export_to_json("data/seed.db", "web-app/public/data.json")
```

**Run after database is built:**
```bash
python3 scripts/export_for_web.py
```

**Expected output:** `web-app/public/data.json` (~500KB)

### M1.4: Create Data Loading Module

`web-app/src/data/loader.ts`:

```typescript
export interface Point {
  id: number;
  text: string;
  topic: string;
  x: number;
  y: number;
}

export interface Neighbor {
  id: number;
  similarity: number;
}

export interface LatentSpaceData {
  items: Point[];
  edges: Record<number, Neighbor[]>;
  metadata: {
    num_items: number;
    num_edges: number;
  };
}

export async function loadLatentSpaceData(): Promise<LatentSpaceData> {
  const response = await fetch('/data.json');
  if (!response.ok) {
    throw new Error(`Failed to load data: ${response.statusText}`);
  }
  return response.json();
}

// Get neighbors for a specific point
export function getNeighbors(
  data: LatentSpaceData,
  pointId: number,
  limit: number = 12
): Point[] {
  const neighborIds = data.edges[pointId] || [];
  return neighborIds
    .slice(0, limit)
    .map(n => data.items.find(item => item.id === n.id))
    .filter((item): item is Point => item !== undefined);
}
```

### M1.5: Create Type Definitions

`web-app/src/types/index.ts`:

```typescript
export type { Point, Neighbor, LatentSpaceData } from '../data/loader';

export interface CameraState {
  position: [number, number, number];
  target: [number, number, number];
}

export interface SelectionState {
  selectedId: number | null;
  hoveredId: number | null;
}

export const TOPIC_COLORS: Record<string, string> = {
  philosophy: '#FF6B6B',
  cognition: '#4ECDC4',
  ai_ml: '#45B7D1',
  language: '#96CEB4',
  affect: '#FFEAA7',
  social: '#DFE6E9',
  systems: '#A29BFE'
};
```

---

## Phase 3: 3D Visualization (2-3 hours)

### M1.6: Create Point Cloud Component

`web-app/src/components/PointCloud.tsx`:

```typescript
import { useMemo, useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Point } from '../types';
import { TOPIC_COLORS } from '../types';

interface PointCloudProps {
  points: Point[];
  selectedId: number | null;
  hoveredId: number | null;
  onPointClick: (id: number) => void;
  onPointHover: (id: number | null) => void;
}

export function PointCloud({
  points,
  selectedId,
  hoveredId,
  onPointClick,
  onPointHover
}: PointCloudProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  // Create geometry and colors
  const { geometry, colors, positions } = useMemo(() => {
    const geometry = new THREE.SphereGeometry(0.05, 8, 8);
    const colors = new Float32Array(points.length * 3);
    const positions: [number, number, number][] = [];

    points.forEach((point, i) => {
      // Position (x, y from UMAP, z = 0 for 2D)
      positions.push([point.x, point.y, 0]);

      // Color based on topic
      const color = new THREE.Color(TOPIC_COLORS[point.topic] || '#999999');
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    });

    return { geometry, colors, positions };
  }, [points]);

  // Update instance matrices
  useFrame(() => {
    if (!meshRef.current) return;

    const dummy = new THREE.Object3D();
    points.forEach((point, i) => {
      dummy.position.set(point.x, point.y, 0);

      // Scale up selected/hovered points
      const scale = point.id === selectedId ? 2.0 :
                   point.id === hoveredId ? 1.5 : 1.0;
      dummy.scale.set(scale, scale, scale);

      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  // Raycasting for click/hover
  const handlePointerMove = (e: THREE.Event) => {
    if (e.instanceId !== undefined) {
      onPointHover(points[e.instanceId].id);
    } else {
      onPointHover(null);
    }
  };

  const handleClick = (e: THREE.Event) => {
    if (e.instanceId !== undefined) {
      onPointClick(points[e.instanceId].id);
    }
  };

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, undefined, points.length]}
      onClick={handleClick}
      onPointerMove={handlePointerMove}
      onPointerOut={() => onPointHover(null)}
    >
      <meshBasicMaterial vertexColors />
    </instancedMesh>
  );
}
```

### M1.7: Create Scene Component

`web-app/src/components/Scene.tsx`:

```typescript
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { PointCloud } from './PointCloud';
import { Point } from '../types';

interface SceneProps {
  points: Point[];
  selectedId: number | null;
  hoveredId: number | null;
  onPointClick: (id: number) => void;
  onPointHover: (id: number | null) => void;
}

export function Scene({
  points,
  selectedId,
  hoveredId,
  onPointClick,
  onPointHover
}: SceneProps) {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
        {/* Lighting */}
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />

        {/* Grid helper */}
        <Grid args={[20, 20]} cellColor="#666" sectionColor="#999" />

        {/* Point cloud */}
        <PointCloud
          points={points}
          selectedId={selectedId}
          hoveredId={hoveredId}
          onPointClick={onPointClick}
          onPointHover={onPointHover}
        />

        {/* Camera controls */}
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={2}
          maxDistance={50}
        />
      </Canvas>
    </div>
  );
}
```

---

## Phase 4: UI Components (1-2 hours)

### M1.8: Create Details Panel

`web-app/src/components/DetailsPanel.tsx`:

```typescript
import { Point } from '../types';
import './DetailsPanel.css';

interface DetailsPanelProps {
  selectedPoint: Point | null;
  neighbors: Point[];
  onClose: () => void;
  onSelectNeighbor: (id: number) => void;
}

export function DetailsPanel({
  selectedPoint,
  neighbors,
  onClose,
  onSelectNeighbor
}: DetailsPanelProps) {
  if (!selectedPoint) return null;

  return (
    <div className="details-panel">
      <div className="details-header">
        <h2>Selected Concept</h2>
        <button onClick={onClose}>×</button>
      </div>

      <div className="details-content">
        <div className="concept-text">{selectedPoint.text}</div>
        <div className="concept-meta">
          <span className="topic-tag" style={{
            backgroundColor: TOPIC_COLORS[selectedPoint.topic]
          }}>
            {selectedPoint.topic}
          </span>
        </div>

        <h3>Nearest Neighbors</h3>
        <ul className="neighbors-list">
          {neighbors.map(neighbor => (
            <li
              key={neighbor.id}
              onClick={() => onSelectNeighbor(neighbor.id)}
              className="neighbor-item"
            >
              <span className="neighbor-text">{neighbor.text}</span>
              <span className="neighbor-topic">{neighbor.topic}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
```

`web-app/src/components/DetailsPanel.css`:

```css
.details-panel {
  position: fixed;
  top: 20px;
  right: 20px;
  width: 400px;
  max-height: 80vh;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
  overflow: hidden;
  z-index: 1000;
}

.details-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid #eee;
}

.details-header h2 {
  margin: 0;
  font-size: 18px;
}

.details-header button {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  padding: 0 8px;
}

.details-content {
  padding: 16px;
  overflow-y: auto;
  max-height: calc(80vh - 60px);
}

.concept-text {
  font-size: 16px;
  line-height: 1.5;
  margin-bottom: 12px;
}

.topic-tag {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
  color: white;
  font-weight: 600;
  margin-bottom: 20px;
}

.neighbors-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.neighbor-item {
  padding: 12px;
  margin-bottom: 8px;
  background: #f5f5f5;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
}

.neighbor-item:hover {
  background: #e8e8e8;
}

.neighbor-text {
  display: block;
  font-size: 14px;
  margin-bottom: 4px;
}

.neighbor-topic {
  display: block;
  font-size: 12px;
  color: #666;
}
```

### M1.9: Create Search Component

`web-app/src/components/SearchBar.tsx`:

```typescript
import { useState } from 'react';
import { Point } from '../types';
import './SearchBar.css';

interface SearchBarProps {
  points: Point[];
  onSelectPoint: (id: number) => void;
}

export function SearchBar({ points, onSelectPoint }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Point[]>([]);

  const handleSearch = (value: string) => {
    setQuery(value);

    if (value.length < 2) {
      setResults([]);
      return;
    }

    const filtered = points
      .filter(p =>
        p.text.toLowerCase().includes(value.toLowerCase())
      )
      .slice(0, 10);

    setResults(filtered);
  };

  return (
    <div className="search-bar">
      <input
        type="text"
        placeholder="Search concepts..."
        value={query}
        onChange={(e) => handleSearch(e.target.value)}
      />

      {results.length > 0 && (
        <ul className="search-results">
          {results.map(point => (
            <li
              key={point.id}
              onClick={() => {
                onSelectPoint(point.id);
                setQuery('');
                setResults([]);
              }}
            >
              {point.text}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

---

## Phase 5: Main App Integration (30 min)

### M1.10: Update App.tsx

`web-app/src/App.tsx`:

```typescript
import { useEffect, useState } from 'react';
import { Scene } from './components/Scene';
import { DetailsPanel } from './components/DetailsPanel';
import { SearchBar } from './components/SearchBar';
import { loadLatentSpaceData, getNeighbors, LatentSpaceData, Point } from './data/loader';
import './App.css';

export default function App() {
  const [data, setData] = useState<LatentSpaceData | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load data on mount
  useEffect(() => {
    loadLatentSpaceData()
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="loading">Loading latent space...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  if (!data) {
    return null;
  }

  const selectedPoint = selectedId
    ? data.items.find(p => p.id === selectedId) || null
    : null;

  const neighbors = selectedId
    ? getNeighbors(data, selectedId)
    : [];

  return (
    <div className="app">
      <SearchBar
        points={data.items}
        onSelectPoint={setSelectedId}
      />

      <Scene
        points={data.items}
        selectedId={selectedId}
        hoveredId={hoveredId}
        onPointClick={setSelectedId}
        onPointHover={setHoveredId}
      />

      <DetailsPanel
        selectedPoint={selectedPoint}
        neighbors={neighbors}
        onClose={() => setSelectedId(null)}
        onSelectNeighbor={setSelectedId}
      />

      <div className="legend">
        <h3>Topics</h3>
        {Object.entries(TOPIC_COLORS).map(([topic, color]) => (
          <div key={topic} className="legend-item">
            <span className="legend-color" style={{ backgroundColor: color }} />
            <span>{topic}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## Testing Checklist

- [ ] Data loads without errors
- [ ] 2000 points render smoothly (>50 fps)
- [ ] Can pan/zoom with mouse
- [ ] Clicking a point shows details panel
- [ ] Clicking neighbor navigates to that point
- [ ] Search finds concepts
- [ ] Colors match topics
- [ ] Hover highlights points

---

## Performance Optimization

If rendering is slow:

1. **Reduce point count for testing:** Load first 500 items only
2. **Use LOD (Level of Detail):** Show simpler geometry when zoomed out
3. **Implement frustum culling:** Only render visible points
4. **Use GPU instancing:** Already implemented with InstancedMesh

---

## Deployment

Once working locally:

```bash
npm run build
# Outputs to web-app/dist/

# Deploy to GitHub Pages, Vercel, Netlify, etc.
```

---

## Next Steps After MVP

1. **P1: Audio feedback** - Add Web Audio API pitch mapping
2. **P1: Interpolation** - Scrubber between two selected points
3. **P2: 3D toggle** - Switch between 2D and 3D view
4. **P2: Filters** - Filter by topic, search refinement
5. **P2: Export** - Download exploration path as JSON

---

## Estimated Timeline

| Phase | Time | Total |
|-------|------|-------|
| Setup | 30 min | 0.5h |
| Data Loading | 1-2h | 2.5h |
| 3D Visualization | 2-3h | 5.5h |
| UI Components | 1-2h | 7.5h |
| Integration & Testing | 1h | 8.5h |

**Total: ~8-10 hours of focused development**

**Calendar time: 2-3 days** at sustainable pace
