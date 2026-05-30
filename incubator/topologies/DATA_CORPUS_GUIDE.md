# Data & Corpus Guide

## 1 Seed Corpus
- 5–10 k snippets (~100 tokens each): philosophy, art, concepts.
- Format: CSV `id,text,topic`.

## 2 Embedding Pipeline
```bash
python scripts/embed_corpus.py \
  --model all-MiniLM-L6-v2 \
  --input data/corpus.csv \
  --embeddings data/embeddings.npy \
  --coords data/coords_umap.npy \
  --umap
```

## 3 Projection (example Python)
```python
import umap, numpy as np
X = np.load('data/embeddings.npy')
coords = umap.UMAP(n_neighbors=15,min_dist=0.1).fit_transform(X)
np.save('data/coords_umap.npy', coords)
```

## 4 Build SQLite seed
```bash
python scripts/build_seed_db.py \
  --corpus data/corpus.csv \
  --embeddings data/embeddings.npy \
  --coords data/coords_umap.npy \
  --out data/seed.db \
  --knn 12
```

## 5 Adding User Text (on-device)
- Embed locally (Core ML / TFLite)
- Find k nearest neighbors in embedding space
- Place via IDW interpolation into 2D map
