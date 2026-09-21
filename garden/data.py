"""Dataset loading + fixed-encoder embeddings for text-attributed graphs.

Datasets (all with raw node text + ground-truth class labels):
  cora, citeseer, wikics  -- HF Graph-COM/Text-Attributed-Graphs
  arxiv                            -- ogbn-arxiv + OGB titleabs.tsv
  photo, history                   -- CS-TAG benchmark (Yan et al. 2023), HF Sherirto/CSTAG: Amazon co-purchase
                                      graphs with product reviews / descriptions; isolated nodes are dropped
"""
import csv
import gzip
import os
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

warnings.filterwarnings("ignore")


def _clean_edges(edge_index, n):
    u, v = edge_index[0], edge_index[1]
    keep = u != v
    u, v = u[keep], v[keep]
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    e = np.unique(np.stack([lo, hi], 1), axis=0)
    assert e.max() < n
    return e


def _load_hf(name):
    import torch
    d = DATA / name
    data = torch.load(d / "processed_data.pt", weights_only=False)
    texts = torch.load(d / "raw_texts.pt", weights_only=False)
    with open(d / "categories.csv") as f:
        label_names = [r[0] for r in csv.reader(f)][1:]
    y = data.y.numpy().astype(int)
    n = len(texts)
    edges = _clean_edges(data.edge_index.numpy(), n)
    if name == "wikics":
        texts = [t.replace("feature node. wikipedia entry name: ", "Title: ", 1)
                  .replace(". entry content: ", ". Content: ", 1) for t in texts]
    return dict(name=name, n=n, edges=edges, y=y, texts=list(texts), label_names=label_names)


def _load_arxiv():
    d = DATA / "arxiv" / "arxiv"
    edges = np.loadtxt(gzip.open(d / "raw" / "edge.csv.gz", "rt"), delimiter=",", dtype=np.int64)
    y = np.loadtxt(gzip.open(d / "raw" / "node-label.csv.gz", "rt"), dtype=np.int64)
    n = len(y)
    edges = _clean_edges(edges.T, n)
    nid2pid = {}
    with gzip.open(d / "mapping" / "nodeidx2paperid.csv.gz", "rt") as f:
        r = csv.reader(f); next(r)
        for a, b in r:
            nid2pid[int(a)] = int(b)
    pid2text = {}
    with open(DATA / "arxiv" / "titleabs.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            pid2text[pid] = f"Title: {parts[1]}\nAbstract: {parts[2]}"
    texts = [pid2text.get(nid2pid[i], "") for i in range(n)]
    label_names = []
    with gzip.open(d / "mapping" / "labelidx2arxivcategeory.csv.gz", "rt") as f:
        r = csv.reader(f); next(r)
        for idx, cat in r:
            label_names.append(cat.replace("arxiv cs ", "cs.").upper().replace("CS.", "cs."))
    return dict(name="arxiv", n=n, edges=edges, y=y, texts=texts, label_names=label_names)


def _load_cstag(name):
    """CS-TAG Amazon graphs: edges from the CSV 'neighbour' lists (symmetrised); isolated nodes dropped."""
    import ast
    import pandas as pd
    fn = {"photo": "Photo", "history": "History"}[name]
    df = pd.read_csv(DATA / name / f"{fn}.csv")
    assert (df["node_id"].values == np.arange(len(df))).all()
    src, dst = [], []
    for i, s_ in enumerate(df["neighbour"].values):
        for j in (ast.literal_eval(s_) if isinstance(s_, str) else []):
            src.append(i); dst.append(int(j))
    e = _clean_edges(np.array([src, dst], dtype=np.int64), len(df))
    keep = np.zeros(len(df), bool); keep[np.unique(e)] = True
    remap = -np.ones(len(df), dtype=np.int64); remap[keep] = np.arange(keep.sum())
    e = remap[e]
    sub = df[keep].reset_index(drop=True)
    y = sub["label"].values.astype(int)
    names = {}
    for lab, cat in zip(sub["label"].values, sub["category"].values):
        names.setdefault(int(lab), cat)
    label_names = [names[i] for i in range(int(y.max()) + 1)]
    texts = [str(t) if isinstance(t, str) else "" for t in sub["text"].values]
    return dict(name=name, n=len(sub), edges=e, y=y, texts=texts, label_names=label_names, n_isolated_dropped=int((~keep).sum()))


def load(name):
    if name == "arxiv":
        return _load_arxiv()
    if name in ("photo", "history"):
        return _load_cstag(name)
    return _load_hf(name)


_encoder = None
import threading
_enc_lock = threading.Lock()


def get_encoder():
    global _encoder
    with _enc_lock:
      if _encoder is None:
          from sentence_transformers import SentenceTransformer
          import torch
          dev = "mps" if torch.backends.mps.is_available() else "cpu"
          _encoder = SentenceTransformer(ENCODER, device=dev)
    return _encoder


def encode(texts, batch_size=256):
    enc = get_encoder()
    with _enc_lock:
      E = enc.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=len(texts) > 5000)
    return E.astype(np.float32)


def node_embeddings(ds):
    """Cached L2-normalised document embeddings (n x d)."""
    p = DATA / ds["name"] / "emb_minilm.npy"
    if p.exists():
        return np.load(p)
    E = encode(ds["texts"])
    np.save(p, E)
    return E


def adjacency(ds):
    """Neighbour lists as a list of int arrays, and degrees."""
    n, e = ds["n"], ds["edges"]
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    deg = np.bincount(src, minlength=n)
    ptr = np.concatenate([[0], np.cumsum(deg)])
    nbrs = [dst[ptr[i]:ptr[i + 1]] for i in range(n)]
    return nbrs, deg
