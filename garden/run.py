"""Usage: python run.py --dataset cora --methods louvain,sn,rdm-centroid,garden --lam 0.5 --sem-norm --seed 0
Appends metrics to results/<dataset>.jsonl and saves the partition to results/preds/."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from data import load, node_embeddings, adjacency
from methods import louvain_init, run_louvain_k, run_sn_centroid, run_rdm, run_rdm3
from evaluate import evaluate
from llm import Describer

DOMAIN = {
    "cora": "citation network of machine-learning papers (title + abstract)",
    "citeseer": "citation network of computer-science papers",
    "wikics": "hyperlink network of Wikipedia articles on computer-science topics",
    "arxiv": "citation network of arXiv computer-science papers (title + abstract)",
    "photo": "co-purchase network of Amazon photography products (customer reviews)",
    "history": "co-purchase network of history books on Amazon (book descriptions and reviews)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", default="louvain,sn,rdm-centroid,garden")
    ap.add_argument("--lam", type=float, nargs="+", default=[0.5])
    ap.add_argument("--K", type=int, default=None, help="number of communities (default: #classes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm-model", default="claude-haiku-4-5")
    ap.add_argument("--max-iter", type=int, default=6)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--shuffle-text", action="store_true", help="randomly permute documents across nodes (uninformative text)")
    ap.add_argument("--no-reliability", action="store_true")
    ap.add_argument("--sem-norm", action="store_true", help="rescale Q_sem so that lambda=0.5 balances per-node move gains")
    ap.add_argument("--adj", type=float, default=0.5, help="parameter of the description-adjusted centroid (gamma for garden-shift, tau for garden-wmean)")
    ap.add_argument("--w3", default=None, help="rdm3 weights 'wc,wd' (graph gets 1-wc-wd)")
    ap.add_argument("--lazy", type=float, default=0.9, help="GARDEN lazy-update Jaccard threshold (1.0 = always regenerate unless identical, 0.0 = never regenerate)")
    args = ap.parse_args()

    ds = load(args.dataset)
    if args.shuffle_text:
        perm = np.random.default_rng(123).permutation(ds["n"])
        ds["texts"] = [ds["texts"][i] for i in perm]
        ds["shuffled"] = True
    K = args.K or len(ds["label_names"])
    print(f"{args.dataset}: n={ds['n']} m={len(ds['edges'])} classes={len(ds['label_names'])} K={K}")
    t0 = time.time()
    E = node_embeddings(ds)
    if args.shuffle_text:
        E = E[perm]
    print(f"embeddings {E.shape} ({time.time()-t0:.1f}s)")
    nbrs, deg = adjacency(ds)

    t0 = time.time()
    init = louvain_init(ds, seed=args.seed)
    print(f"Louvain init: K0={init[1]} ({time.time()-t0:.1f}s)")
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True); (out / "preds").mkdir(exist_ok=True)
    fout = open(out / f"{args.dataset}.jsonl", "a")

    def record(name, pred, info, lam=None, t=None):
        ev = evaluate(ds, deg, pred)
        row = dict(dataset=args.dataset, method=name, lam=lam, seed=args.seed, K_target=K, time=t, lazy=args.lazy,
                   tag=args.tag, **ev, **{k: v for k, v in info.items() if k != "descriptions"})
        print(f"  {name:14s} lam={lam}  K={ev['K']} NMI={ev['NMI']:.4f} ARI={ev['ARI']:.4f} "
              f"ACC={ev['ACC']:.4f} purity={ev['purity']:.4f} Q={ev['Q']:.4f}  ({t:.1f}s)")
        if "descriptions" in info:
            row["descriptions"] = info["descriptions"]
            # what GT classes each community mostly contains
            comp = {}
            for c in np.unique(pred):
                ys = ds["y"][pred == c]
                top = np.bincount(ys, minlength=len(ds["label_names"]))
                comp[int(c)] = {ds["label_names"][i]: int(top[i]) for i in np.argsort(-top)[:3] if top[i] > 0}
            row["composition"] = comp
        row["shuffled"] = args.shuffle_text
        fout.write(json.dumps(row) + "\n"); fout.flush()
        tagsfx = "" if args.tag in ("", "final") else f"_{args.tag}"
        np.save(out / "preds" / f"{args.dataset}_{name}_{lam}_{args.seed}{tagsfx}{'_shuf' if args.shuffle_text else ''}.npy", pred)

    for meth in args.methods.split(","):
        if meth == "louvain":
            t0 = time.time(); pred, info = run_louvain_k(ds, nbrs, deg, K, init, args.verbose)
            record(meth, pred, info, t=time.time() - t0)
        elif meth == "sn":
            t0 = time.time(); pred, info = run_sn_centroid(ds, nbrs, deg, E, K, init, verbose=args.verbose)
            record(meth, pred, info, t=time.time() - t0)
        elif meth.startswith("rdm-"):
            for lam in args.lam:
                t0 = time.time()
                pred, info = run_rdm(ds, nbrs, deg, E, K, init, lam, proto=meth[4:], max_iter=args.max_iter, reliability=not args.no_reliability, normalize=args.sem_norm, verbose=args.verbose)
                record(meth, pred, info, lam=lam, t=time.time() - t0)
        elif meth == "garden":
            for lam in args.lam:
                t0 = time.time()
                desc = Describer(ds, nbrs, E, args.llm_model, DOMAIN[args.dataset], lazy_jaccard=args.lazy, seed=args.seed, verbose=True)
                pred, info = run_rdm(ds, nbrs, deg, E, K, init, lam, proto="llm", describer=desc, max_iter=args.max_iter, reliability=not args.no_reliability, normalize=args.sem_norm, verbose=args.verbose)
                record(meth, pred, info, lam=lam, t=time.time() - t0)
        elif meth in ("garden-shift", "garden-wmean", "garden-wmargin", "garden-trim", "garden-selfw", "garden-cagree", "garden-cexp", "garden-dexp"):
            for lam in args.lam:
                t0 = time.time()
                desc = Describer(ds, nbrs, E, args.llm_model, DOMAIN[args.dataset], lazy_jaccard=args.lazy, seed=args.seed, verbose=True)
                pred, info = run_rdm(ds, nbrs, deg, E, K, init, lam, proto=meth.split("-")[1], describer=desc, max_iter=args.max_iter,
                                     reliability=not args.no_reliability, normalize=args.sem_norm, verbose=args.verbose, adj_param=args.adj)
                info["adj"] = args.adj
                record(meth, pred, info, lam=lam, t=time.time() - t0)
        elif meth == "rdm3":
            wc, wd = map(float, args.w3.split(","))
            t0 = time.time()
            desc = Describer(ds, nbrs, E, args.llm_model, DOMAIN[args.dataset], lazy_jaccard=args.lazy, seed=args.seed, verbose=True)
            pred, info = run_rdm3(ds, nbrs, deg, E, K, init, wc, wd, desc, max_iter=args.max_iter, reliability=not args.no_reliability, verbose=args.verbose)
            record(meth, pred, info, lam=wd, t=time.time() - t0)
        else:
            raise ValueError(meth)


if __name__ == "__main__":
    main()
