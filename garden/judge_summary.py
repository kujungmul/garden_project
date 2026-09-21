"""Summarise the corrected judge audit (results/judge_audit.jsonl) against the v1 numbers.
Usage: python judge_summary.py"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "results"
METHODS = ["louvain", "sn", "rdm-centroid", "garden"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


rows = [json.loads(l) for l in open(ROOT / "judge_audit.jsonl")]
by = defaultdict(list)
for r in rows:
    by[(r["dataset"], r["method"], r["pass"])].append(r)
byq = defaultdict(dict)
for r in rows:
    byq[(r["dataset"], r["pred_file"], r["cid"])][r["pass"]] = r

print("A. query accounting per dataset x method (all passes pooled)")
print(f"{'dataset':9s} {'method':13s} {'queries':>7s} {'valid':>6s} {'empty':>6s} {'trunc':>6s} {'retried':>7s} {'unresolved':>10s}")
for ds in ["cora", "citeseer", "wikics", "arxiv"]:
    for m in METHODS:
        rs = [r for k, v in by.items() if k[0] == ds and k[1] == m for r in v]
        if not rs:
            continue
        print(f"{ds:9s} {m:13s} {len(rs):7d} {sum(r['valid'] for r in rs):6d} {sum(r['empty'] for r in rs):6d} "
              f"{sum(r['stop_reason']=='max_tokens' for r in rs):6d} {sum(r['attempts']>1 for r in rs):7d} {sum(r['judged'] is None for r in rs):10d}")

print("\nB. judge accuracy: v1 as reported (failures counted wrong) | v1 excluding failures | v2 p1 (corrected, Opus 5) [95% Wilson CI over communities] | seed-level mean+-std")
v1 = defaultdict(list)
for ds in ["cora", "citeseer", "wikics", "arxiv"]:
    for r in map(json.loads, open(ROOT / f"desc_{ds}.jsonl")):
        v1[(ds, r["method"])].append(r)
for ds in ["cora", "citeseer", "wikics", "arxiv"]:
    print(f"-- {ds}")
    for m in METHODS:
        rs = by.get((ds, m, "p1"), [])
        if not rs:
            continue
        ok = [r["correct"] for r in rs if r["correct"] is not None]
        k, n = sum(ok), len(ok); lo, hi = wilson(k, n)
        per_seed = defaultdict(list)
        for r in rs:
            if r["correct"] is not None:
                per_seed[r["seed"]].append(r["correct"])
        sm = [np.mean(v) for v in per_seed.values()]
        v1rep = np.mean([x["judge_acc"] for x in v1[(ds, m)]])
        v1ex = np.mean([np.mean([d["judged"] == d["majority"] for d in x["descriptions"].values() if d["judged"] is not None]) for x in v1[(ds, m)]])
        print(f"   {m:13s} v1 {v1rep:.3f} | v1-excl {v1ex:.3f} | v2 {k/n:.3f} [{lo:.3f},{hi:.3f}] n={n} | seeds {np.mean(sm):.3f}+-{np.std(sm):.3f}")

print("\nC. robustness: agreement of p1 with p3 (repeat), p2 (shuffled class order), p4 (Sonnet 5); and accuracy under each")
for ds in ["cora", "citeseer", "wikics", "arxiv"]:
    print(f"-- {ds}")
    for m in METHODS:
        qs = [q for k, q in byq.items() if k[0] == ds and "p1" in q and q["p1"]["method"] == m]
        if not qs:
            continue
        out = []
        for ps in ["p3", "p2", "p4"]:
            pairs = [(q["p1"], q[ps]) for q in qs if ps in q and q["p1"]["judged"] is not None and q[ps]["judged"] is not None]
            if not pairs:
                out.append(f"{ps}: n/a"); continue
            agr = np.mean([a["judged"] == b["judged"] for a, b in pairs])
            acc = np.mean([b["correct"] for _, b in pairs])
            out.append(f"{ps}: agree {agr:.3f} acc {acc:.3f} (n={len(pairs)})")
        # majority vote over the three Opus passes
        votes = []
        for q in qs:
            js = [q[p]["judged"] for p in ("p1", "p2", "p3") if p in q and q[p]["judged"] is not None]
            if len(js) == 3:
                top = max(set(js), key=js.count); votes.append(top == q["p1"]["majority"])
        mv = f"majority-vote acc {np.mean(votes):.3f} (n={len(votes)})" if votes else ""
        print(f"   {m:13s} " + " | ".join(out) + " | " + mv)
