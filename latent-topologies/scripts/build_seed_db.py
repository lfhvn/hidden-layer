#!/usr/bin/env python3
"""
Build a SQLite seed database from corpus CSV, embeddings (.npy), and coords (.npy).
Also computes a simple kNN graph using cosine similarity (no FAISS dependency).
Requires: numpy, pandas.
Example:
  python scripts/build_seed_db.py \
    --corpus data/corpus.csv \
    --embeddings data/embeddings.npy \
    --coords data/coords_umap.npy \
    --out data/seed.db \
    --knn 12
"""
import argparse, sqlite3, numpy as np, pandas as pd, os

def cosine_sim_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, 1e-8, None)
    return Xn @ Xn.T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--coords", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--knn", type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(args.corpus)
    X = np.load(args.embeddings)
    coords = np.load(args.coords)
    assert len(df) == X.shape[0] == coords.shape[0], "Mismatched lengths"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    con = sqlite3.connect(args.out)
    cur = con.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS item (
      id INTEGER PRIMARY KEY,
      text TEXT NOT NULL,
      topic TEXT
    );
    CREATE TABLE IF NOT EXISTS embedding (
      item_id INTEGER REFERENCES item(id),
      dim INTEGER NOT NULL,
      vec BLOB NOT NULL,
      PRIMARY KEY(item_id)
    );
    CREATE TABLE IF NOT EXISTS coord2d (
      item_id INTEGER REFERENCES item(id) PRIMARY KEY,
      x REAL NOT NULL,
      y REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS edge (
      src INTEGER,
      dst INTEGER,
      weight REAL,
      PRIMARY KEY(src, dst)
    );
    """)

    cur.executemany("INSERT INTO item(id,text,topic) VALUES (?,?,?)",
                    [(int(r.id), str(r.text), str(r.topic)) for r in df.itertuples(index=False)])

    dim = X.shape[1]
    for i, vec in enumerate(X, start=1):
        cur.execute("INSERT INTO embedding(item_id, dim, vec) VALUES (?,?,?)",
                    (i, dim, memoryview(vec.astype(np.float32).tobytes())))

    for i, (x,y) in enumerate(coords, start=1):
        cur.execute("INSERT INTO coord2d(item_id, x, y) VALUES (?,?,?)", (i, float(x), float(y)))

    sims = cosine_sim_matrix(X)
    n = sims.shape[0]
    for i in range(n):
        idx = np.argpartition(-sims[i], range(1, args.knn+1))[1:args.knn+1]
        for j in idx:
            if i == j:
                continue
            cur.execute("INSERT OR REPLACE INTO edge(src,dst,weight) VALUES (?,?,?)",
                        (i+1, j+1, float(sims[i, j])))

    con.commit()
    con.close()
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
