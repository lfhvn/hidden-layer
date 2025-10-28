#!/usr/bin/env python3
"""
Embed a CSV corpus of short texts using a local sentence embedding model,
optionally compute a UMAP projection, and save outputs.
Requires: sentence-transformers, numpy, umap-learn (optional), pandas.
Example:
  python scripts/embed_corpus.py \
    --model all-MiniLM-L6-v2 \
    --input data/corpus.csv \
    --embeddings data/embeddings.npy \
    --coords data/coords_umap.npy \
    --umap
"""
import argparse, os, numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--input", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--coords", default=None)
    ap.add_argument("--umap", action="store_true")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    df = pd.read_csv(args.input)
    texts = df["text"].astype(str).tolist()

    model = SentenceTransformer(args.model)
    X = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    os.makedirs(os.path.dirname(args.embeddings), exist_ok=True)
    np.save(args.embeddings, X)

    if args.umap:
        try:
            import umap
        except ImportError:
            raise SystemExit("Install umap-learn to compute projection: pip install umap-learn")
        coords = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine").fit_transform(X)
        if args.coords is None:
            raise SystemExit("--coords path required when --umap is set")
        np.save(args.coords, coords)

if __name__ == "__main__":
    main()
