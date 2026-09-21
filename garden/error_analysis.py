"""Error analysis of the corrected graph+centroid partitions (Cora, CiteSeer):
which nodes are misassigned, are they boundary nodes, and could a description-based
correction rule fix them WITHOUT breaking currently-correct nodes?

A node is 'wrong' when the majority GT class of its community != its own GT class.
Candidate communities for a node = {own} u {communities of its neighbours} (what local move can reach).
Signals per node (restricted to candidates):
    cent   nearest centroid          desc   nearest post-hoc description
    graph  community with most neighbours (ties -> current)
Correction rules (applied to EVERY node, so both fixes and breaks are counted):
    R1 move to desc          R2 move to graph          R3 move to cent
    R4 move where desc == graph != current             R5 move where desc == cent != current
    R6 move where desc == graph == cent != current
Outputs results/error_analysis.md (tables) and results/error_nodes_{ds}_{seed}.csv (per node).
Usage: python error_analysis.py cora citeseer
"""
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

from data import load, node_embeddings, adjacency, encode

ROOT = Path(__file__).resolve().parent.parent / "results"


def analyse(ds, seed, out_rows):
    d = load(ds); y = np.asarray(d["y"]); E = node_embeddings(d); nbrs, deg = adjacency(d)
    pred = np.load(ROOT / "preds" / f"{ds}_rdm-centroid_0.5_{seed}_audit.npy"); K = pred.max() + 1
    desc = next(r for r in map(json.loads, open(ROOT / f"desc_audit_{ds}.jsonl")) if r["method"] == "rdm-centroid" and r["tag"] == "audit" and r["seed"] == seed)
    texts = {int(c): v["text"] for c, v in desc["descriptions"].items()}
    C = np.stack([E[pred == c].mean(0) for c in range(K)]); C /= np.linalg.norm(C, axis=1, keepdims=True)
    Z = encode([texts.get(c, "") for c in range(K)]); Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    Sc = E @ C.T; Sd = E @ Z.T
    maj = np.array([np.bincount(y[pred == c], minlength=y.max() + 1).argmax() for c in range(K)])
    names = d["label_names"]
    n = len(y); rows = []
    for v in range(n):
        cur = int(pred[v]); nb = pred[nbrs[v]]
        cand = sorted(set(nb.tolist()) | {cur})
        cnt = Counter(nb.tolist())
        graph = max(cand, key=lambda c: (cnt.get(c, 0), c == cur))
        cent = max(cand, key=lambda c: Sc[v, c]); dsc = max(cand, key=lambda c: Sd[v, c])
        sc = sorted([Sc[v, c] for c in cand], reverse=True); margin = sc[0] - sc[1] if len(sc) > 1 else 1.0
        right_avail = any(maj[c] == y[v] for c in cand)
        rows.append(dict(seed=seed, v=v, y=int(y[v]), y_name=names[y[v]], cur=cur, cur_maj=names[maj[cur]], correct=bool(maj[cur] == y[v]),
                         n_cand=len(cand), right_available=bool(right_avail), margin=float(margin), deg=int(deg[v]),
                         frac_nbr_own=float(cnt.get(cur, 0) / max(len(nb), 1)),
                         cent=cent, desc=dsc, graph=graph, cent_ok=bool(maj[cent] == y[v]), desc_ok=bool(maj[dsc] == y[v]), graph_ok=bool(maj[graph] == y[v]),
                         text=d["texts"][v][:160].replace("\n", " ")))
    out_rows.extend(rows)
    with open(ROOT / f"error_nodes_{ds}_{seed}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def rule_stats(rows, pick):
    """pick(row) -> community to move to, or None. Returns fixes, breaks, net, moved."""
    fixes = breaks = moved = 0
    for r in rows:
        t = pick(r)
        if t is None or t == r["cur"]:
            continue
        moved += 1
        ok_after = {r["cent"]: r["cent_ok"], r["desc"]: r["desc_ok"], r["graph"]: r["graph_ok"]}[t]
        if not r["correct"] and ok_after:
            fixes += 1
        elif r["correct"] and not ok_after:
            breaks += 1
    return fixes, breaks, fixes - breaks, moved


def main():
    md = ["# Error analysis: misassigned nodes in the corrected graph+centroid partitions", "",
          "Partition: `rdm-centroid`, tag `audit`, seeds 0–4. 'wrong' = majority GT class of the node's community ≠ its GT class. "
          "Candidates = own community ∪ neighbours' communities. Signals restricted to candidates. Per-node data: `error_nodes_{ds}_{seed}.csv`.", ""]
    for ds in sys.argv[1:]:
        allrows = []
        for seed in range(5):
            analyse(ds, seed, allrows)
        n = len(allrows) / 5
        wrong = [r for r in allrows if not r["correct"]]; right = [r for r in allrows if r["correct"]]
        md.append(f"## {ds} (n={int(n)} nodes, 5 seeds pooled)\n")
        md.append(f"- wrong nodes: {len(wrong)/5:.0f} per seed ({len(wrong)/len(allrows):.1%})")
        md.append(f"- of the wrong nodes, a community of the right class is among the candidates for {np.mean([r['right_available'] for r in wrong]):.1%} "
                  f"(the rest cannot be fixed by any local move; their class has no adjacent community)")
        md.append(f"- boundary check — centroid margin (top-1 − top-2 among candidates): wrong nodes median {np.median([r['margin'] for r in wrong]):.3f}, "
                  f"correct nodes median {np.median([r['margin'] for r in right]):.3f}; wrong nodes with margin in the lowest quartile of all nodes: "
                  f"{np.mean([r['margin'] <= np.quantile([x['margin'] for x in allrows], 0.25) for r in wrong]):.1%}")
        md.append(f"- fraction of neighbours in own community: wrong nodes {np.mean([r['frac_nbr_own'] for r in wrong]):.2f}, correct nodes {np.mean([r['frac_nbr_own'] for r in right]):.2f}")
        md.append("")
        md.append("**Which signal points to the right class?** (fraction of nodes whose pick has the node's GT class)\n")
        md.append("| nodes | centroid | description | graph vote | current |")
        md.append("|---|---|---|---|---|")
        for lab, rs in (("wrong (fixable: right class available)", [r for r in wrong if r["right_available"]]), ("correct", right)):
            md.append(f"| {lab} (n={len(rs)/5:.0f}/seed) | {np.mean([r['cent_ok'] for r in rs]):.3f} | {np.mean([r['desc_ok'] for r in rs]):.3f} | {np.mean([r['graph_ok'] for r in rs]):.3f} | {np.mean([r['correct'] for r in rs]):.3f} |")
        md.append("")
        md.append("**Correction rules applied to all nodes** (per seed averages; net = fixes − breaks; a rule helps only if net > 0)\n")
        md.append("| rule | moved | fixes | breaks | net | net as % of wrong |")
        md.append("|---|---|---|---|---|---|")
        rules = [("R1 move to nearest description", lambda r: r["desc"]), ("R2 move to graph vote", lambda r: r["graph"]),
                 ("R3 move to nearest centroid", lambda r: r["cent"]),
                 ("R4 desc == graph, both ≠ current", lambda r: r["desc"] if r["desc"] == r["graph"] else None),
                 ("R5 desc == centroid, both ≠ current", lambda r: r["desc"] if r["desc"] == r["cent"] else None),
                 ("R6 desc == graph == centroid ≠ current", lambda r: r["desc"] if r["desc"] == r["graph"] == r["cent"] else None),
                 ("R4' as R4 but only low-margin nodes (bottom quartile)", lambda r: r["desc"] if (r["desc"] == r["graph"] and r["margin"] <= q25) else None)]
        q25 = np.quantile([x["margin"] for x in allrows], 0.25)
        for lab, pick in rules:
            f, b, net, mv = rule_stats(allrows, pick)
            md.append(f"| {lab} | {mv/5:.0f} | {f/5:.0f} | {b/5:.0f} | {net/5:+.0f} | {net/len(wrong):+.1%} |")
        md.append("")
        # class confusion of wrong nodes
        conf = Counter((r["y_name"], r["cur_maj"]) for r in wrong)
        md.append("**Most frequent confusions (GT class → class of assigned community), per seed**\n")
        md.append("| GT class | assigned to | count/seed |"); md.append("|---|---|---|")
        for (a, b), c in conf.most_common(8):
            md.append(f"| {a} | {b} | {c/5:.0f} |")
        md.append("")
    (ROOT / "error_analysis.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
