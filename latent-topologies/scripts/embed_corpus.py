#!/usr/bin/env python3
"""
Embed a CSV corpus of short texts using a local sentence embedding model,
optionally compute a UMAP projection, and save outputs.

Features:
- Robust logging and validation
- Checkpointing for large datasets
- Memory-efficient batch processing
- UMAP model persistence for new point placement

Requires: sentence-transformers, numpy, umap-learn (optional), pandas.

Example:
  python scripts/embed_corpus.py \
    --model all-MiniLM-L6-v2 \
    --input data/corpus.csv \
    --embeddings data/embeddings.npy \
    --coords data/coords_umap.npy \
    --umap \
    --save-umap-model data/umap_model.pkl
"""
import argparse
import os
import sys
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('embed_corpus.log')
    ]
)
logger = logging.getLogger(__name__)


def validate_corpus(df: pd.DataFrame) -> bool:
    """Validate corpus DataFrame structure and content."""
    logger.info("Validating corpus...")

    # Check required columns
    if "text" not in df.columns:
        logger.error("Corpus missing required 'text' column")
        return False

    # Check for empty texts
    empty_count = df["text"].isna().sum()
    if empty_count > 0:
        logger.warning(f"Found {empty_count} empty text entries - will be filtered")

    # Check text lengths
    df["text_len"] = df["text"].astype(str).str.len()
    logger.info(f"Text length stats: min={df['text_len'].min()}, "
                f"max={df['text_len'].max()}, mean={df['text_len'].mean():.1f}")

    # Warn about very short or very long texts
    too_short = (df["text_len"] < 10).sum()
    too_long = (df["text_len"] > 1000).sum()
    if too_short > 0:
        logger.warning(f"{too_short} texts are very short (< 10 chars)")
    if too_long > 0:
        logger.warning(f"{too_long} texts are very long (> 1000 chars)")

    logger.info(f"✓ Corpus validation passed: {len(df)} entries")
    return True


def generate_embeddings(texts: list, model_name: str, batch_size: int = 64,
                       checkpoint_path: Optional[str] = None) -> np.ndarray:
    """Generate embeddings with optional checkpointing."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Check for existing checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Found checkpoint at {checkpoint_path}, loading...")
        checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
        if checkpoint.get("model_name") == model_name and checkpoint.get("num_texts") == len(texts):
            logger.info("✓ Checkpoint valid, resuming from saved embeddings")
            return checkpoint["embeddings"]
        else:
            logger.warning("Checkpoint invalid (model or corpus changed), regenerating...")

    logger.info(f"Encoding {len(texts)} texts in batches of {batch_size}...")
    X = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save checkpoint
    if checkpoint_path:
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        np.save(checkpoint_path, {
            "embeddings": X,
            "model_name": model_name,
            "num_texts": len(texts)
        })

    logger.info(f"✓ Generated embeddings with shape {X.shape}")
    return X


def compute_umap_projection(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1,
                           metric: str = "cosine", random_state: int = 42,
                           save_model_path: Optional[str] = None) -> tuple:
    """Compute UMAP projection and optionally save the model for new point placement."""
    try:
        import umap
    except ImportError:
        raise SystemExit("Install umap-learn to compute projection: pip install umap-learn")

    logger.info(f"Computing UMAP projection (n_neighbors={n_neighbors}, min_dist={min_dist})...")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True
    )

    coords = reducer.fit_transform(X)
    logger.info(f"✓ UMAP projection complete: {coords.shape}")

    # Save UMAP model for placing new points
    if save_model_path:
        logger.info(f"Saving UMAP model to {save_model_path}")
        with open(save_model_path, 'wb') as f:
            pickle.dump(reducer, f)
        logger.info("✓ UMAP model saved (use for placing new points)")

    return coords, reducer


def main():
    ap = argparse.ArgumentParser(
        description="Generate embeddings and UMAP projections for text corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    ap.add_argument("--input", required=True, help="Input CSV file with 'text' column")
    ap.add_argument("--embeddings", required=True, help="Output path for embeddings (.npy)")
    ap.add_argument("--coords", default=None, help="Output path for UMAP coordinates (.npy)")
    ap.add_argument("--umap", action="store_true", help="Compute UMAP projection")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding")
    ap.add_argument("--checkpoint", default=None, help="Checkpoint path for resuming")
    ap.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    ap.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    ap.add_argument("--save-umap-model", default=None, help="Save UMAP model for new point placement")
    args = ap.parse_args()

    try:
        # Load and validate corpus
        logger.info(f"Loading corpus from {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} entries")

        if not validate_corpus(df):
            sys.exit(1)

        # Filter and prepare texts
        df = df[df["text"].notna()].copy()
        texts = df["text"].astype(str).tolist()
        logger.info(f"Processing {len(texts)} valid texts")

        # Generate embeddings
        X = generate_embeddings(texts, args.model, args.batch_size, args.checkpoint)

        # Save embeddings
        os.makedirs(os.path.dirname(args.embeddings) or ".", exist_ok=True)
        logger.info(f"Saving embeddings to {args.embeddings}")
        np.save(args.embeddings, X)

        # Compute UMAP if requested
        if args.umap:
            if args.coords is None:
                logger.error("--coords path required when --umap is set")
                sys.exit(1)

            coords, umap_model = compute_umap_projection(
                X,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
                save_model_path=args.save_umap_model
            )

            logger.info(f"Saving UMAP coordinates to {args.coords}")
            np.save(args.coords, coords)

        logger.info("✓ All done!")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
