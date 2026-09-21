"""Failure taxonomy of description prototypes (audit F).

For every described community of the rdm-centroid and garden partitions we compare, on the
community's own members, nearest-prototype class accuracy under the post-hoc description vs
under the member centroid, and flag the observable failure indicators below. A community is a
'failure case' when description accuracy is >= 15 points below centroid accuracy.

Indicators (all computable from the encoder, the partition and GT labels; judge flags are joined
from results/judge_audit.jsonl, pass p1, when available):
  hetero      community purity < 0.5 (no single concept can represent it)
  broad       description contrast (mean s to members - mean s to random non-members) in the bottom
              quartile of all communities of the dataset (generic description)
  confusable  another description in the same partition has cos >= 0.6 with this one AND that
              community absorbs >= 50% of this community's lost members
  repdocs     majority class of the 10 representative documents != majority class of the community
  fine        lost members go to a community whose majority class is the SAME as this one's
              (a within-class split: the description cannot express the finer distinction)
  encoder     the judge maps the description to the right class, but retrieval is still poor
              (concept right, encoder geometry wrong)
  graphtext   >= 50% of the members that the description misassigns have GT label != community
              majority (their text belongs elsewhere; the graph put them here)
Usage: python failure_taxonomy.py cora citeseer wikics arxiv [--source v2|audit]
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from data import load, node_embeddings, adjacency, encode
from llm import representative_docs

ROOT = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--source", default="v2", help="v2 = pre-audit desc_{ds}.jsonl, audit = desc_audit_{ds}.jsonl")
    ap.add_argument("--gap", type=float, default=0.15)
    args = ap.parse_args()
    judge = defaultdict(dict)
    if (ROOT / "judge_audit.jsonl").exists():
        for r in map(json.loads, open(ROOT / "judge_audit.jsonl")):
            if r["pass"] == "p1" and r["judged"] is not None:
                judge[r["pred_file"]][r["cid"]] = (r["judged"] == r["majority"])
    grand = Counter(); grand_n = 0; examples = []
    for ds in args.datasets:
        d = load(ds); y = np.asarray(d["y"]); E = node_embeddings(d); nbrs, deg = adjacency(d)
        rows = [json.loads(l) for l in open(ROOT / (f"desc_{ds}.jsonl" if args.source == "v2" else f"desc_audit_{ds}.jsonl"))]
        rows = [r for r in rows if r["method"] in ("rdm-centroid", "garden") and r.get("tag", "final") in ("final", "audit")]
        comm_rows = []
        for r in rows:
            pred = np.load(ROOT / "preds" / r["pred_file"]); K = pred.max() + 1
            desc = r["descriptions"]; cids = sorted(int(c) for c in desc)
            Z = encode([desc[str(c)]["text"] for c in cids]); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            C = np.stack([E[pred == c].mean(0) for c in cids]); C /= np.linalg.norm(C, axis=1, keepdims=True)
            Sd = (1 + E @ Z.T) / 2; Sc = (1 + E @ C.T) / 2
            maj = np.array([np.bincount(y[pred == c], minlength=y.max() + 1).argmax() for c in cids])
            nd, nc = Sd.argmax(1), Sc.argmax(1)
            rng = np.random.default_rng(0)
            neg = rng.choice(len(y), size=min(512, len(y)), replace=False)
            contrast = np.array([Sd[pred == c, i].mean() - Sd[neg, i].mean() for i, c in enumerate(cids)])
            ZZ = Z @ Z.T; np.fill_diagonal(ZZ, -1)
            for i, c in enumerate(cids):
                m = pred == c
                if m.sum() < 20:
                    continue
                acc_d = (maj[nd[m]] == y[m]).mean(); acc_c = (maj[nc[m]] == y[m]).mean()
                lost = m & (nd != i)                                    # members retrieved by another description
                dest = Counter(nd[lost].tolist())
                top_dest, top_cnt = (dest.most_common(1)[0] if dest else (None, 0))
                docs = representative_docs(np.flatnonzero(m), nbrs, pred, E, d["texts"])
                rep_maj = np.bincount(y[[v for v, _ in docs]], minlength=y.max() + 1).argmax()
                purity = (y[m] == maj[i]).mean()
                jud = judge.get(r["pred_file"], {}).get(c, None)
                misassigned_alien = ((y[lost] != maj[i]).mean() if lost.any() else 0.0)
                flags = dict(
                    hetero=purity < 0.5,
                    broad=False,                                        # filled after quartiles are known
                    confusable=(top_dest is not None and ZZ[i, top_dest] >= 0.6 and top_cnt >= 0.5 * lost.sum()),
                    repdocs=rep_maj != maj[i],
                    fine=(top_dest is not None and maj[top_dest] == maj[i] and top_cnt >= 0.5 * lost.sum()),
                    encoder=(jud is True and acc_d < acc_c - args.gap),
                    graphtext=misassigned_alien >= 0.5,
                )
                comm_rows.append(dict(ds=ds, method=r["method"], seed=r["seed"], cid=c, size=int(m.sum()), acc_d=acc_d, acc_c=acc_c,
                                      gap=acc_c - acc_d, contrast=contrast[i], purity=purity, judge=jud, text=desc[str(c)]["text"],
                                      majority=desc[str(c)]["majority"], top_dest_text=(desc[str(cids[top_dest])]["text"] if top_dest is not None else None),
                                      top_dest_maj=(desc[str(cids[top_dest])]["majority"] if top_dest is not None else None), flags=flags))
        q1 = np.quantile([x["contrast"] for x in comm_rows], 0.25)
        for x in comm_rows:
            x["flags"]["broad"] = x["contrast"] <= q1
        fails = [x for x in comm_rows if x["gap"] >= args.gap]
        better = [x for x in comm_rows if x["gap"] <= -args.gap]
        print(f"\n==== {ds}: {len(comm_rows)} described communities (size>=20) in {len(rows)} partitions; "
              f"description worse than centroid by >= {args.gap:.0%}: {len(fails)} ({len(fails)/len(comm_rows):.0%}); better by >= {args.gap:.0%}: {len(better)} ({len(better)/len(comm_rows):.0%})")
        print(f"   mean member NN-class acc: description {np.mean([x['acc_d'] for x in comm_rows]):.3f}  centroid {np.mean([x['acc_c'] for x in comm_rows]):.3f}")
        cnt = Counter(); base = Counter()
        for x in comm_rows:
            for k, v in x["flags"].items():
                base[k] += bool(v)
        for x in fails:
            for k, v in x["flags"].items():
                cnt[k] += bool(v); grand[k] += bool(v)
        grand_n += len(fails)
        print("   indicator          among failures      among all communities")
        for k in ["hetero", "broad", "confusable", "fine", "repdocs", "encoder", "graphtext"]:
            print(f"   {k:12s}   {cnt[k]:4d} / {len(fails):4d} ({(cnt[k]/max(len(fails),1)):5.0%})      {base[k]:4d} / {len(comm_rows):4d} ({base[k]/len(comm_rows):5.0%})")
        nof = sum(1 for x in fails if not any(x["flags"].values()))
        print(f"   failures with no indicator: {nof}")
        for x in sorted(fails, key=lambda x: -x["gap"])[:4]:
            examples.append(x)
    print("\n==== worst examples (largest centroid - description gap) ====")
    for x in sorted(examples, key=lambda x: -x["gap"])[:12]:
        fl = ",".join(k for k, v in x["flags"].items() if v) or "-"
        print(f"[{x['ds']} {x['method']} s{x['seed']} c{x['cid']} n={x['size']}] gap {x['gap']:.2f} (desc {x['acc_d']:.2f} cent {x['acc_c']:.2f}) purity {x['purity']:.2f} judge={x['judge']} flags={fl}")
        print(f"     desc: {x['text']}  [majority: {x['majority']}]")
        if x["top_dest_text"]:
            print(f"     lost members go to: {x['top_dest_text']}  [majority: {x['top_dest_maj']}]")


if __name__ == "__main__":
    main()
