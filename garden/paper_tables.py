"""Emit LaTeX-ready numbers for the paper from results/*.jsonl (corrected code, 5 seeds).
Usage: python paper_tables.py > ../results/paper_tables.txt"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment

from data import load

ROOT = Path(__file__).resolve().parent.parent / "results"
DS = ["cora", "citeseer", "wikics", "arxiv"]
ROWS = [("audit", "louvain", "Louvain"), ("audit", "sn", "SN-modularity"), ("audit", "rdm-centroid", "SemMod (centroid)"),
        ("audit", "garden", "SemMod (description)"), ("cexp", "garden-cexp", "SemMod + non-local (centroid)"),
        ("dexp", "garden-dexp", "SemMod + non-local (description)"), ("cagree", "garden-cagree", "GARDEN (agreement)"),
        ("wmean3", "garden-wmean", "desc-weighted centroid tau=3"), ("selfw3", "garden-selfw", "self-weighted centroid tau=3")]
cl = defaultdict(list); de = defaultdict(list)
for ds in DS:
    for r in map(json.loads, open(ROOT / f"{ds}.jsonl")):
        if r.get("lam") in (None, 0.5) and not r.get("shuffled"):
            cl[(ds, r["tag"], r["method"])].append(r)
    for r in map(json.loads, open(ROOT / f"desc_audit_{ds}.jsonl")):
        de[(ds, r.get("tag", "final"), r["method"])].append(r)


def ms(v, d=3):
    v = [x for x in v if x is not None]
    return f"{np.mean(v):.{d}f}$\\pm${np.std(v):.{d}f}" if v else "--"


print("% ---- clustering quality ----")
for key in ["NMI", "ARI", "ACC", "purity", "Q"]:
    print(f"% {key}")
    for t, m, lab in ROWS:
        print(f"{lab} & " + " & ".join(ms([r[key] for r in cl[(ds, t, m)]]) for ds in DS) + r" \\")
print("% ---- describability ----")
for key in ["retrieval_heldout", "judge_acc", "Qsem"]:
    print(f"% {key}")
    for t, m, lab in ROWS:
        print(f"{lab} & " + " & ".join(ms([r.get(key) for r in de[(ds, t, m)]]) for ds in DS) + r" \\")
print("% ---- paired deltas vs centroid (mean, wins) ----")
for key in ["NMI", "ACC", "purity"]:
    print(f"% d{key}")
    for t, m, lab in ROWS[3:]:
        cells = []
        for ds in DS:
            cen = {r["seed"]: r[key] for r in cl[(ds, "audit", "rdm-centroid")]}
            dd = [r[key] - cen[r["seed"]] for r in cl[(ds, t, m)] if r["seed"] in cen]
            cells.append(f"{np.mean(dd):+.3f} ({sum(x > 0 for x in dd)}/{len(dd)})" if dd else "--")
        print(f"{lab} & " + " & ".join(cells) + r" \\")

print("% ---- structure: trapped errors, seeds/followers, connectivity ----")
def align(p, q):
    K = max(p.max(), q.max()) + 1; M = np.zeros((K, K)); np.add.at(M, (p, q), 1)
    r, c = linear_sum_assignment(-M); mp = dict(zip(c, r)); return np.array([mp[x] for x in q])
for ds in DS:
    d = load(ds); y = np.asarray(d["y"]); n = d["n"]
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(map(tuple, d["edges"]))
    nbrs = [np.array(list(G.neighbors(v)), dtype=int) for v in range(n)]
    st = defaultdict(list)
    for s in range(5):
        pb = np.load(ROOT / "preds" / f"{ds}_rdm-centroid_0.5_{s}_audit.npy"); pc = np.load(ROOT / "preds" / f"{ds}_garden-cagree_0.5_{s}_cagree.npy")
        majb = np.array([np.bincount(y[pb == c], minlength=y.max() + 1).argmax() for c in range(pb.max() + 1)])
        majc = np.array([np.bincount(y[pc == c], minlength=y.max() + 1).argmax() for c in range(pc.max() + 1)])
        wrong = majb[pb] != y
        # trapped: no neighbour community of the right class
        fixable = np.array([any(majb[c] == y[v] for c in set(pb[nbrs[v]].tolist())) for v in range(n)])
        st["wrong"].append(wrong.mean()); st["trapped_of_wrong"].append((wrong & ~fixable).sum() / wrong.sum())
        st["nbr_own_wrong"].append(np.mean([(pb[nbrs[v]] == pb[v]).mean() for v in np.flatnonzero(wrong) if len(nbrs[v])]))
        pca = align(pb, pc); moved = np.flatnonzero(pca != pb)
        iso = np.array([len(nbrs[v]) > 0 and not (pc[nbrs[v]] == pc[v]).any() for v in moved])
        st["moved"].append(len(moved) / n); st["seed_frac"].append(iso.mean() if len(iso) else 0)
        st["moved_right_before"].append((majb[pb[moved]] == y[moved]).mean()); st["moved_right_after"].append((majc[pc[moved]] == y[moved]).mean())
        for tag, p in (("centroid", pb), ("cagree", pc)):
            iso_all = np.array([len(nbrs[v]) > 0 and not (p[nbrs[v]] == p[v]).any() for v in range(n)])
            frag = sum(len(x) for c in range(p.max() + 1) for x in sorted(nx.connected_components(G.subgraph(np.flatnonzero(p == c))), key=len, reverse=True)[1:])
            st[f"iso_{tag}"].append(iso_all.mean()); st[f"frag_{tag}"].append(frag / n)
    a = {k: np.mean(v) for k, v in st.items()}
    print(f"{ds} & wrong {a['wrong']:.1%} & trapped-of-wrong {a['trapped_of_wrong']:.1%} & nbr-own(wrong) {a['nbr_own_wrong']:.2f} & moved {a['moved']:.1%} & seeds {a['seed_frac']:.0%} "
          f"& moved-right {a['moved_right_before']:.1%}->{a['moved_right_after']:.1%} & semantic-only centroid {a['iso_centroid']:.1%} cagree {a['iso_cagree']:.1%} & fragmented {a['frag_centroid']:.1%}->{a['frag_cagree']:.1%} & components {nx.number_connected_components(G)}")
print("% ---- Cora exploratory variants ----")
for (d, t, m), rs in sorted(cl.items()):
    if d == "cora" and (m.startswith("garden-") or m == "rdm3") and t not in ("audit",):
        ds_ = de.get(("cora", t, m), [])
        print(f"{m} {t} & n={len(rs)} & {ms([r['NMI'] for r in rs])} & {ms([r['ACC'] for r in rs])} & {ms([r['purity'] for r in rs])} & {ms([r.get('retrieval_heldout') for r in ds_])} & {ms([r.get('judge_acc') for r in ds_])} \\\\")
print("% ---- runtime (s) and llm calls ----")
for t, m, lab in ROWS:
    print(f"{lab} & " + " & ".join((f"{np.mean([r['time'] for r in cl[(ds, t, m)]]):.0f} / {np.mean([r.get('llm_calls', 0) or 0 for r in cl[(ds, t, m)]]):.0f}" if cl[(ds, t, m)] else "--") for ds in DS) + r" \\")
print("% ---- reliability gate under shuffled text (tag shuffle2, seed 0): gate outcome, Louvain NMI, GARDEN NMI, identical to SemMod? ----")
LAB = {"cora": "Cora", "citeseer": "CiteSeer", "wikics": "WikiCS", "arxiv": "ogbn-arxiv"}
for ds in DS:
    sh = {r["method"]: r for r in map(json.loads, open(ROOT / f"{ds}.jsonl")) if r.get("tag") == "shuffle2" and r.get("shuffled")}
    if not {"louvain", "rdm-centroid", "garden-cagree"} <= set(sh):
        print(f"{LAB[ds]} & -- (shuffle2 runs missing) \\\\"); continue
    g = sh["garden-cagree"]; gate = g["reliability"]
    outcome = "fail in all blocks" if all(x["lam_eff"] == 0 for x in gate) else f"pass in {sum(x['lam_eff'] > 0 for x in gate)}/{len(gate)} blocks"
    pc = np.load(ROOT / "preds" / f"{ds}_rdm-centroid_0.5_0_shuffle2_shuf.npy"); pg = np.load(ROOT / "preds" / f"{ds}_garden-cagree_0.5_0_shuffle2_shuf.npy")
    rho = "; ".join(f"{x['acc']:.3f}/{x['chance']:.3f}" for x in gate)
    print(f"{LAB[ds]} & {outcome} & {0 if outcome.startswith('fail') else '0.5'} & {sh['louvain']['NMI']:.3f} & {g['NMI']:.3f} \\\\  % rho_obs/rho_null per block: {rho}; GARDEN partition == SemMod: {np.array_equal(pc, pg)}")
