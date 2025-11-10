# Claude Development Guide - Latent Topologies

## Project Overview

**Latent Topologies** is a web-based application for exploring AI embedding spaces through interactive visualization. It transforms high-dimensional semantic representations into an explorable visual landscape.

**Core Concept**: Make latent space tangible - see relationships between concepts, explore semantic neighborhoods, and understand how AI models organize knowledge.

---

## Quick Start

### For Development

```bash
cd representations/latent-space/topologies

# 1. Generate data (if not already done)
python3 scripts/generate_corpus.py --output data/corpus_2k.csv --size 2000
python3 scripts/embed_corpus.py --input data/corpus_2k.csv --embeddings data/embeddings_2k.npy --coords data/coords_2k.npy --umap
python3 scripts/build_seed_db.py --corpus data/corpus_2k.csv --embeddings data/embeddings_2k.npy --coords data/coords_2k.npy --out data/seed.db

# 2. Run web app (once implemented)
cd web-app
npm install
npm run dev
```

---

## Architecture

### Data Pipeline (Python)

```
[Seed Concepts] → [Generate Corpus] → [Embed] → [UMAP] → [SQLite]
     ↓                  ↓                 ↓         ↓          ↓
   SEED_CONCEPTS   corpus_2k.csv   embeddings  coords_2k   seed.db
  (7 domains)      (1995 items)     (384D)      (2D)      (kNN graph)
```

### Web App (React + Three.js)

```
[SQLite] → [Data Layer] → [Three.js Scene] → [UI Components]
                              ↓
                        Point Cloud
                        Camera Controls
                        Raycasting
                              ↓
                        [User Interactions]
                         - Pan/Zoom
                         - Select
                         - Search
                         - Filter
```

---

## Current Status (MVP Planning Phase)

### ✅ Completed

**M0: Data Pipeline**
- ✅ M0.1: Generated 2000-item corpus (285 items × 7 domains)
- ✅ M0.2: Fixed corpus generation script (sampling with replacement)
- 🔄 M0.3: Installing dependencies (in progress)
- ⏳ M0.4: Generate embeddings + UMAP projection
- ⏳ M0.5: Build SQLite database

**Documentation**
- ✅ PRD, Technical Plan, UX Storyboard
- ✅ Audio/Haptic specifications (P1)
- ✅ Integration plan with harness
- ✅ This CLAUDE.md guide

### ⏳ In Progress

- Installing Python dependencies (PyTorch, sentence-transformers, UMAP)
- Preparing to generate embeddings

### 📋 Next Steps

**M1: Web App MVP** (target: 3-4 days)
1. Initialize Vite + React + TypeScript project
2. Setup Three.js + react-three-fiber
3. Load SQLite data (via API or bundled JSON)
4. Render 2D/3D point cloud
5. Implement camera controls (orbit, pan, zoom)
6. Add raycasting for point selection
7. Build details panel UI
8. Add search and filtering

**P1: Audio + Haptics** (post-MVP)
- Web Audio API for pitch-from-distance mapping
- Vibration API for boundary crossing (limited browser support)

---

## Development Decisions

### Web App vs Mobile

**Decision**: **Build web app first**

**Rationale**:
- Instant accessibility (share URL, no install)
- Faster iteration (React + browser devtools)
- Better for research studies (send link to participants)
- Three.js ecosystem is mature
- Audio/haptics (P1 anyway) can be added later
- Can port to React Native later if needed

### Corpus Size

**Decision**: **2000 items for MVP**

**Rationale**:
- 500 too sparse to see interesting clusters
- 2000 provides rich semantic structure (285 per domain)
- UMAP computation still fast (<5 min)
- Sufficient for meaningful exploration
- Can scale to 5K+ post-MVP if needed

### Rendering Strategy

**Options Considered**:
1. **react-three-fiber + drei** (3D library)
2. **@shopify/react-native-skia** (2D, mobile-optimized)
3. **Plain Three.js** (no React wrapper)
4. **D3.js** (2D only, data viz focused)

**Decision**: **react-three-fiber + drei**

**Rationale**:
- Declarative React API (easier to maintain)
- Can do both 2D and 3D
- OrbitControls, raycasting built-in
- Active ecosystem
- Good performance for 2000 points

---

## Data Schema

### Corpus CSV
```csv
id,text,topic
1,"justice: Fairness in distribution...",philosophy
2,"being: The fundamental nature...",philosophy
```

### SQLite Database
```sql
-- Core items
CREATE TABLE item (
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL,
  topic TEXT
);

-- High-D embeddings (384D from all-MiniLM-L6-v2)
CREATE TABLE embedding (
  item_id INTEGER REFERENCES item(id),
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL
);

-- 2D UMAP coordinates for visualization
CREATE TABLE coord2d (
  item_id INTEGER REFERENCES item(id),
  x REAL NOT NULL,
  y REAL NOT NULL
);

-- kNN graph (for nearest neighbor lookup)
CREATE TABLE edge (
  src INTEGER REFERENCES item(id),
  dst INTEGER REFERENCES item(id),
  weight REAL NOT NULL  -- cosine similarity
);

-- Clusters (optional, for coloring/filtering)
CREATE TABLE cluster (
  item_id INTEGER REFERENCES item(id),
  cluster_id INTEGER NOT NULL
);
```

---

## Key Algorithms

### 1. UMAP Projection

**Purpose**: Reduce 384D embeddings to 2D while preserving local structure

**Parameters** (current):
```python
n_neighbors=15   # Local neighborhood size
min_dist=0.1     # Minimum distance between points
metric="cosine"  # Cosine similarity (for text embeddings)
```

**Why these values**:
- `n_neighbors=15`: Balances local vs global structure
- `min_dist=0.1`: Allows some overlap but not too tight
- Can tune based on visual inspection

### 2. kNN Graph Construction

**Purpose**: Pre-compute nearest neighbors for instant lookup

**Parameters**:
```python
k=12  # Number of neighbors per point
```

**Algorithm**: scikit-learn NearestNeighbors with cosine metric

**Why**:
- User taps a concept → instantly show 12 nearest neighbors
- Avoids expensive cosine similarity computation at runtime

### 3. Interpolation (for scrubber feature)

**Linear interpolation in embedding space**:
```python
emb_interp = (1 - t) * emb_a + t * emb_b
# Find nearest corpus item to interpolated embedding
nearest_idx = argmax(cosine_similarity(emb_interp, corpus_embeddings))
```

---

## Research Integration with Harness

### Connection Points

1. **Corpus Generation**: Use harness multi-agent strategies (debate, self-consistency) to generate diverse concepts

2. **Embedding Comparison**: Use harness experiment tracker to compare models
   - all-MiniLM-L6-v2 (90MB, fast)
   - all-mpnet-base-v2 (420MB, quality)
   - e5-base-v2 (alternative)

3. **Annotation Quality**: Use LLM-as-judge from harness to validate user annotations

4. **Experiment Tracking**: Version all corpus generation runs using harness tracker

**See**: `INTEGRATION.md` for detailed workflows

---

## Development Workflow

### Daily Development Loop

```bash
# 1. Activate environment
source venv/bin/activate  # (if using venv)

# 2. Make changes to scripts
# Edit generate_corpus.py, embed_corpus.py, etc.

# 3. Test changes
python3 scripts/generate_corpus.py --output data/test.csv --size 100
python3 scripts/embed_corpus.py --input data/test.csv --embeddings data/test.npy

# 4. Validate
python3 -c "import numpy as np; X = np.load('data/test.npy'); print(X.shape)"

# 5. Commit when stable
git add scripts/ data/corpus_2k.csv
git commit -m "Update corpus generation with X"
```

### Adding New Concepts

**Option 1: Manual** (curated, high-quality)
```python
# Edit SEED_CONCEPTS in scripts/generate_corpus.py
SEED_CONCEPTS = {
    "new_domain": [
        ("concept1", "Definition 1"),
        ("concept2", "Definition 2"),
    ],
    # ... existing domains
}
```

**Option 2: LLM-generated** (diverse, scale)
```bash
# Use --use-llm flag (requires harness)
python3 scripts/generate_corpus.py \
  --output data/corpus_expanded.csv \
  --size 5000 \
  --use-llm \
  --provider ollama \
  --model llama3.2:latest
```

---

## Testing & Validation

### Data Pipeline Tests

```bash
# 1. Generate small corpus
python3 scripts/generate_corpus.py --output data/test_corpus.csv --size 100 --seed 42

# 2. Embed
python3 scripts/embed_corpus.py --input data/test_corpus.csv --embeddings data/test_emb.npy --model all-MiniLM-L6-v2

# 3. Validate embedding shape
python3 << EOF
import numpy as np
X = np.load('data/test_emb.npy')
assert X.shape == (100, 384), f"Expected (100, 384), got {X.shape}"
print("✓ Embeddings valid")
EOF

# 4. Build database
python3 scripts/build_seed_db.py --corpus data/test_corpus.csv --embeddings data/test_emb.npy --coords data/test_coords.npy --out data/test.db

# 5. Query database
sqlite3 data/test.db << EOF
.mode column
SELECT COUNT(*) as num_items FROM item;
SELECT COUNT(*) as num_edges FROM edge;
.quit
EOF
```

### Expected Outputs

```
✓ Corpus: 1995 items, 7 domains
✓ Embeddings: (1995, 384) shape
✓ Coordinates: (1995, 2) shape
✓ Database: ~2MB, 1995 items, ~23940 edges (1995 × 12)
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Corpus generation | <1 sec | For 2000 items without LLM |
| Embedding generation | 2-3 min | First run (downloads model) |
| UMAP projection | 2-3 min | For 2000 items |
| Database build | <30 sec | Including kNN computation |
| Web app load | <2 sec | Including data fetch |
| Rendering FPS | >50 fps | For 2000 points |
| Selection latency | <100ms | Raycasting + UI update |

---

## Troubleshooting

### Python Dependencies Won't Install

**Issue**: `ModuleNotFoundError` or `ImportError`

**Solution**:
```bash
# Ensure requirements.txt exists
cat requirements.txt

# Install with verbose output
pip install -r requirements.txt --verbose

# If UMAP fails on ARM/M1/M2/M4:
brew install llvm libomp
pip install --no-binary umap-learn umap-learn
```

### UMAP Projection Looks Weird

**Issue**: Points clustered too tightly or too spread out

**Solution**:
```bash
# Try different parameters
python3 scripts/embed_corpus.py \
  --input data/corpus_2k.csv \
  --embeddings data/embeddings_2k.npy \
  --coords data/coords_2k.npy \
  --umap \
  --n-neighbors 30 \      # Increase for more global structure
  --min-dist 0.01         # Decrease for tighter packing
```

**Typical values**:
- `n_neighbors`: 5-50 (15 is default)
- `min_dist`: 0.0-0.99 (0.1 is default)

### Database Too Large

**Issue**: `seed.db` is >10MB

**Solution**:
```bash
# Check size breakdown
sqlite3 data/seed.db << EOF
SELECT 'items', COUNT(*) FROM item
UNION
SELECT 'embeddings', COUNT(*) FROM embedding
UNION
SELECT 'edges', COUNT(*) FROM edge;
.quit
EOF

# Options:
# 1. Don't store embeddings in DB (use separate .npy file)
# 2. Reduce k in kNN graph (e.g., k=6 instead of k=12)
# 3. Use float16 instead of float32 for embeddings
```

### Web App Not Rendering Points

**Issue**: Blank canvas or console errors

**Check**:
1. Data loaded correctly?
   ```javascript
   console.log('Loaded points:', points.length);
   ```
2. Coordinates in valid range?
   ```javascript
   console.log('X range:', Math.min(...points.map(p => p.x)), Math.max(...points.map(p => p.x)));
   ```
3. Camera positioned correctly?
   ```javascript
   <OrbitControls target={[0, 0, 0]} />
   <PerspectiveCamera position={[0, 0, 10]} />
   ```

---

## Code Style & Conventions

### Python
- Use type hints where helpful
- Logging over print statements
- Argparse for CLI scripts
- Docstrings for functions
- Black formatter (line length 100)

### TypeScript/React
- Functional components with hooks
- Strict TypeScript (`strict: true`)
- ESLint + Prettier
- Component folder structure:
  ```
  src/
    components/
      LatentMap/
        LatentMap.tsx
        LatentMap.module.css
        types.ts
        utils.ts
    ```

---

## Roadmap

### MVP (Week 1-2)
- [x] M0: Data pipeline (corpus, embeddings, UMAP, DB)
- [ ] M1: Web app foundation (React + Three.js)
- [ ] M2: Core interactions (pan, zoom, select, details)
- [ ] M3: Search and filtering

### P1 (Week 3-4)
- [ ] Audio feedback (Web Audio API)
- [ ] Haptic feedback (Vibration API, where supported)
- [ ] Interpolation scrubber
- [ ] Settings panel (audio on/off, sensitivity)

### P2 (Month 2)
- [ ] Creator mode (user input → embed → place on map)
- [ ] Annotations/notes system
- [ ] Export latent walks
- [ ] User study integration

### Future
- [ ] 3D visualization (toggle 2D/3D)
- [ ] Collaborative maps (multiple users)
- [ ] Temporal tracking (how user's exploration changes over time)
- [ ] Mobile app (React Native port)

---

## References

**Internal Docs**:
- `PRD.md` - Product requirements
- `TECH_PLAN.md` - Technical architecture
- `AUDIO_HAPTIC_SPEC.md` - Detailed audio/haptic mappings
- `INTEGRATION.md` - Harness integration workflows
- `SETUP_DEV.md` - Development environment setup

**External**:
- UMAP: https://umap-learn.readthedocs.io/
- Sentence Transformers: https://www.sbert.net/
- Three.js: https://threejs.org/
- react-three-fiber: https://docs.pmnd.rs/react-three-fiber/

---

## Questions While Developing?

**Data pipeline issues**: Check `scripts/` and existing docs
**Harness integration**: See `INTEGRATION.md` and `/harness/README.md`
**Web app architecture**: See `TECH_PLAN.md` and this guide
**Audio/haptic specs**: See `AUDIO_HAPTIC_SPEC.md`

**Philosophy**: Start simple, iterate based on user feedback, maintain reproducibility
