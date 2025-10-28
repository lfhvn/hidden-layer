#!/usr/bin/env python3
"""
Build a SQLite seed database from corpus CSV, embeddings (.npy), and coords (.npy).

Features:
- Efficient kNN computation with batching
- Enhanced schema with indexes and metadata
- Cluster detection and annotation
- Validation and statistics

Requires: numpy, pandas, scikit-learn (for efficient kNN).

Example:
  python scripts/build_seed_db.py \
    --corpus data/corpus.csv \
    --embeddings data/embeddings.npy \
    --coords data/coords_umap.npy \
    --out data/seed.db \
    --knn 12 \
    --detect-clusters
"""
import argparse
import sqlite3
import os
import sys
import logging
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('build_seed_db.log')
    ]
)
logger = logging.getLogger(__name__)


def validate_inputs(corpus_df: pd.DataFrame, embeddings: np.ndarray, coords: np.ndarray) -> bool:
    """Validate input data consistency."""
    logger.info("Validating inputs...")

    if len(corpus_df) != embeddings.shape[0]:
        logger.error(f"Corpus length ({len(corpus_df)}) != embeddings length ({embeddings.shape[0]})")
        return False

    if len(corpus_df) != coords.shape[0]:
        logger.error(f"Corpus length ({len(corpus_df)}) != coords length ({coords.shape[0]})")
        return False

    if coords.shape[1] != 2:
        logger.error(f"Coords should be 2D, got shape {coords.shape}")
        return False

    logger.info(f"✓ Validation passed: {len(corpus_df)} items, "
                f"{embeddings.shape[1]}D embeddings, 2D coords")
    return True


def compute_knn_efficient(X: np.ndarray, k: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute kNN efficiently using scikit-learn's NearestNeighbors.
    Returns (distances, indices) arrays.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        logger.warning("scikit-learn not available, falling back to slow method")
        return compute_knn_fallback(X, k)

    logger.info(f"Computing kNN graph (k={k}) using scikit-learn...")

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.clip(norms, 1e-8, None)

    # k+1 because query point is included
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute', n_jobs=-1)
    nbrs.fit(X_norm)

    distances, indices = nbrs.kneighbors(X_norm)

    # Remove self (first neighbor)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Convert cosine distance to similarity
    similarities = 1 - distances

    logger.info(f"✓ kNN computed: {len(indices)} items × {k} neighbors")
    return similarities, indices


def compute_knn_fallback(X: np.ndarray, k: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback kNN computation (slower but no sklearn dependency)."""
    logger.info(f"Computing kNN graph (k={k}) using fallback method...")

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.clip(norms, 1e-8, None)
    sims = X_norm @ X_norm.T

    n = sims.shape[0]
    indices = np.zeros((n, k), dtype=np.int32)
    similarities = np.zeros((n, k), dtype=np.float32)

    for i in range(n):
        # Get top k+1 (including self), then exclude self
        idx = np.argpartition(-sims[i], range(k+1))[:k+1]
        idx = idx[idx != i][:k]  # Remove self and take k
        indices[i] = idx
        similarities[i] = sims[i, idx]

    logger.info(f"✓ kNN computed: {n} items × {k} neighbors")
    return similarities, indices


def detect_clusters(coords: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """Detect clusters using HDBSCAN or fallback to KMeans."""
    try:
        import hdbscan
        logger.info("Detecting clusters using HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=3)
        labels = clusterer.fit_predict(coords)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"✓ Detected {n_clusters} clusters (noise: {(labels == -1).sum()})")
        return labels
    except ImportError:
        logger.warning("hdbscan not available, skipping cluster detection")
        return np.full(len(coords), -1, dtype=np.int32)


def create_schema(cur: sqlite3.Cursor):
    """Create enhanced database schema with indexes and metadata."""
    logger.info("Creating database schema...")

    cur.executescript("""
    -- Core item table
    CREATE TABLE IF NOT EXISTS item (
      id INTEGER PRIMARY KEY,
      text TEXT NOT NULL,
      topic TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Embedding storage (BLOB for efficiency)
    CREATE TABLE IF NOT EXISTS embedding (
      item_id INTEGER REFERENCES item(id) ON DELETE CASCADE,
      dim INTEGER NOT NULL,
      vec BLOB NOT NULL,
      PRIMARY KEY(item_id)
    );

    -- 2D coordinates for visualization
    CREATE TABLE IF NOT EXISTS coord2d (
      item_id INTEGER REFERENCES item(id) ON DELETE CASCADE PRIMARY KEY,
      x REAL NOT NULL,
      y REAL NOT NULL
    );

    -- kNN graph edges
    CREATE TABLE IF NOT EXISTS edge (
      src INTEGER REFERENCES item(id) ON DELETE CASCADE,
      dst INTEGER REFERENCES item(id) ON DELETE CASCADE,
      weight REAL NOT NULL,
      PRIMARY KEY(src, dst)
    );
    CREATE INDEX IF NOT EXISTS idx_edge_src ON edge(src);
    CREATE INDEX IF NOT EXISTS idx_edge_dst ON edge(dst);
    CREATE INDEX IF NOT EXISTS idx_edge_weight ON edge(weight DESC);

    -- Cluster assignments
    CREATE TABLE IF NOT EXISTS cluster (
      item_id INTEGER REFERENCES item(id) ON DELETE CASCADE PRIMARY KEY,
      cluster_id INTEGER NOT NULL,
      FOREIGN KEY(item_id) REFERENCES item(id)
    );
    CREATE INDEX IF NOT EXISTS idx_cluster_id ON cluster(cluster_id);

    -- Metadata table (for versioning and provenance)
    CREATE TABLE IF NOT EXISTS metadata (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- User notes/annotations (for future use)
    CREATE TABLE IF NOT EXISTS note (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      item_id INTEGER REFERENCES item(id) ON DELETE CASCADE,
      body TEXT NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_note_item ON note(item_id);
    """)

    logger.info("✓ Schema created with indexes")


def main():
    ap = argparse.ArgumentParser(
        description="Build SQLite database from embeddings and coordinates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--corpus", required=True, help="Input CSV corpus file")
    ap.add_argument("--embeddings", required=True, help="Input embeddings (.npy)")
    ap.add_argument("--coords", required=True, help="Input UMAP coordinates (.npy)")
    ap.add_argument("--out", required=True, help="Output SQLite database path")
    ap.add_argument("--knn", type=int, default=12, help="Number of nearest neighbors")
    ap.add_argument("--detect-clusters", action="store_true", help="Detect and annotate clusters")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing database")
    args = ap.parse_args()

    try:
        # Check if output exists
        if os.path.exists(args.out) and not args.overwrite:
            logger.error(f"Output file {args.out} exists. Use --overwrite to replace.")
            sys.exit(1)

        # Load inputs
        logger.info(f"Loading corpus from {args.corpus}")
        df = pd.read_csv(args.corpus)

        logger.info(f"Loading embeddings from {args.embeddings}")
        X = np.load(args.embeddings)

        logger.info(f"Loading coordinates from {args.coords}")
        coords = np.load(args.coords)

        # Validate
        if not validate_inputs(df, X, coords):
            sys.exit(1)

        # Create database
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        con = sqlite3.connect(args.out)
        cur = con.cursor()

        create_schema(cur)

        # Insert items
        logger.info("Inserting items...")
        cur.executemany(
            "INSERT INTO item(id, text, topic) VALUES (?, ?, ?)",
            [(int(r.id), str(r.text), str(r.get('topic', '')))
             for r in df.itertuples(index=False)]
        )

        # Insert embeddings
        logger.info("Inserting embeddings...")
        dim = X.shape[1]
        for i, vec in enumerate(X, start=1):
            cur.execute(
                "INSERT INTO embedding(item_id, dim, vec) VALUES (?, ?, ?)",
                (i, dim, memoryview(vec.astype(np.float32).tobytes()))
            )

        # Insert coordinates
        logger.info("Inserting coordinates...")
        cur.executemany(
            "INSERT INTO coord2d(item_id, x, y) VALUES (?, ?, ?)",
            [(i+1, float(x), float(y)) for i, (x, y) in enumerate(coords)]
        )

        # Compute and insert kNN graph
        similarities, indices = compute_knn_efficient(X, k=args.knn)

        logger.info("Inserting kNN edges...")
        edges = []
        for i in range(len(indices)):
            for j, neighbor_idx in enumerate(indices[i]):
                edges.append((i+1, int(neighbor_idx)+1, float(similarities[i, j])))

        cur.executemany("INSERT INTO edge(src, dst, weight) VALUES (?, ?, ?)", edges)

        # Detect clusters if requested
        if args.detect_clusters:
            labels = detect_clusters(coords)
            logger.info("Inserting cluster assignments...")
            cur.executemany(
                "INSERT INTO cluster(item_id, cluster_id) VALUES (?, ?)",
                [(i+1, int(label)) for i, label in enumerate(labels)]
            )

        # Insert metadata
        logger.info("Inserting metadata...")
        metadata = {
            "schema_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "num_items": str(len(df)),
            "embedding_dim": str(dim),
            "knn": str(args.knn),
            "corpus_file": args.corpus,
            "embeddings_file": args.embeddings,
            "coords_file": args.coords,
        }
        cur.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            metadata.items()
        )

        # Commit and close
        con.commit()

        # Print statistics
        logger.info("\n" + "="*60)
        logger.info("DATABASE STATISTICS")
        logger.info("="*60)
        logger.info(f"Items: {len(df)}")
        logger.info(f"Embedding dimension: {dim}")
        logger.info(f"kNN edges: {len(edges)}")
        logger.info(f"Database size: {os.path.getsize(args.out) / 1024 / 1024:.2f} MB")
        logger.info("="*60)

        con.close()
        logger.info(f"✓ Database written to {args.out}")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
