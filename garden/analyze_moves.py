"""Analysis of the move-level logs written by instrument.py (audit E).
Usage: python analyze_moves.py cora citeseer wikics
Pools all seeds per dataset. Only candidate moves whose source/destination majority classes differ
are 'GT-informative' (moving v changes whether it sits in a community of its own class)."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "results"
EPS = 1e-9


def load(ds):
    logs, accs, ns = [], [], []
    for f in sorted(ROOT.glob(f"moves_{ds}_*.npz")):
        z = np.load(f); L = z["log"]; A = z["accepted"]
        n = len(z["y"])
        acc = set(map(tuple, A[:, [0, 1, 2, 4]].tolist()))          # (phase, pass, v, B)
        acc_flag = np.array([(int(r[0]), int(r[1]), int(r[2]), int(r[4])) in acc for r in L])
        seed = np.full(len(L), int(f.stem.split("_")[-1]))
        logs.append(L); accs.append(acc_flag); ns.append(np.full(len(L), n))
    return np.vstack(logs), np.concatenate(accs), np.concatenate(ns)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def report(ds):
    L, acc, n = load(ds)
    c = dict(zip("phase,pass,v,A,B,dq,dm_desc,dm_cent,mdA,mcA,mdB,mcB,lam_eff,scale_desc,scale_cent,y,majA,majB,sizeA,sizeB".split(","), range(20)))
    y, majA, majB = L[:, c["y"]], L[:, c["majA"]], L[:, c["majB"]]
    gt = (y == majB).astype(int) - (y == majA).astype(int)            # +1 move improves, -1 hurts, 0 uninformative
    dq, dd, dc = L[:, c["dq"]], L[:, c["dm_desc"]], L[:, c["dm_cent"]]
    lam = L[:, c["lam_eff"]]; sd, sc = L[:, c["scale_desc"]], L[:, c["scale_cent"]]
    dQd = lam * sd * dd / n; dQc = lam * sc * dc / n; dQg = (1 - lam) * dq
    active = lam > 0
    print(f"\n==== {ds}: {len(L)} candidate moves ({acc.sum()} accepted), {active.mean()*100:.0f}% evaluated with the semantic term active ====")
    inf = (gt != 0) & active
    print(f"GT-informative candidate moves (majority class of A != B, semantic term active): {inf.sum()}")
    sd_, sc_ = np.sign(dd), np.sign(dc)
    both = inf & (np.abs(dd) > EPS) & (np.abs(dc) > EPS)
    agree = both & (sd_ == sc_); dis = both & (sd_ != sc_)
    print(f"1-2. sign agreement of d_margin_desc vs d_margin_cent: agree {agree.sum()/both.sum()*100:.1f}%  disagree {dis.sum()/both.sum()*100:.1f}%  (of {both.sum()})")
    for name, m in (("agree", agree), ("disagree", dis)):
        kd = ((sd_ == gt) & m).sum(); kc = ((sc_ == gt) & m).sum(); N = m.sum()
        print(f"   on {name:8s}: desc sign matches GT {kd/N*100:.1f}% [{wilson(kd,N)[0]*100:.1f},{wilson(kd,N)[1]*100:.1f}]   cent sign matches GT {kc/N*100:.1f}% [{wilson(kc,N)[0]*100:.1f},{wilson(kc,N)[1]*100:.1f}]   (N={N})")
    # 4. boundary / ambiguous nodes: bins of |d_margin_cent| (small = geometry indifferent) and of the graph gain
    print("4. by |d_margin_cent| quantile bin (small = geometrically ambiguous move):")
    q = np.quantile(np.abs(dc[inf]), [0, .1, .25, .5, .75, 1])
    for lo, hi in zip(q[:-1], q[1:]):
        m = inf & (np.abs(dc) >= lo) & (np.abs(dc) <= hi) & (np.abs(dd) > EPS)
        kd = ((sd_ == gt) & m).sum(); kc = ((sc_ == gt) & m & (np.abs(dc) > EPS)).sum(); N = m.sum()
        print(f"   |dc| in [{lo:.4f},{hi:.4f}]: N={N:6d}  desc-right {kd/N*100:5.1f}%  cent-right {kc/N*100:5.1f}%")
    print("   by graph gain sign (dq<0: graph opposes the move; dq>0: graph favours it):")
    for name, m0 in (("dq<0", dq < 0), ("dq>=0", dq >= 0)):
        m = inf & m0 & (np.abs(dd) > EPS) & (np.abs(dc) > EPS)
        kd = ((sd_ == gt) & m).sum(); kc = ((sc_ == gt) & m).sum(); N = m.sum()
        print(f"   {name:5s}: N={N:6d}  desc-right {kd/N*100:5.1f}%  cent-right {kc/N*100:5.1f}%")
    # 5-6. usefulness vs description reliability: community-level mean own-description margin at that phase
    key = L[:, c["phase"]] * 10**6 + L[:, c["A"]]
    uniq, inv = np.unique(key, return_inverse=True)
    r_comm = np.bincount(inv, weights=L[:, c["mdA"]]) / np.bincount(inv)      # mean own margin of nodes in A at this phase
    rA = r_comm[inv]
    keyB = L[:, c["phase"]] * 10**6 + L[:, c["B"]]
    lookup = dict(zip(uniq.tolist(), r_comm.tolist()))
    rB = np.array([lookup.get(k, np.nan) for k in keyB.tolist()])
    rmin = np.fmin(rA, rB)
    print("5-6. on DISAGREEMENT moves, description correctness vs reliability r = min(mean own-desc margin of A, of B):")
    qs = np.nanquantile(rmin[dis], [0, .25, .5, .75, .9, 1])
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = dis & (rmin >= lo) & (rmin <= hi)
        kd = ((sd_ == gt) & m).sum(); N = m.sum()
        print(f"   r in [{lo:.4f},{hi:.4f}]: N={N:5d}  desc-right {kd/N*100:5.1f}% [{wilson(kd,N)[0]*100:.1f},{wilson(kd,N)[1]*100:.1f}]  (cent-right {100-kd/N*100:5.1f}%)")
    for t in np.nanquantile(rmin[dis], [.5, .75, .9, .95]):
        m = dis & (rmin >= t); kd = ((sd_ == gt) & m).sum(); N = m.sum()
        print(f"   threshold r>={t:.4f}: N={N:5d} desc-right {kd/N*100:5.1f}% [{wilson(kd,N)[0]*100:.1f},{wilson(kd,N)[1]*100:.1f}]")
    # accepted moves: what did the description term actually do?
    a = acc & active & (gt != 0)
    flip_graph = a & (dQg <= 0)                        # graph alone would not have moved v to B
    cent_reject = a & (dQg + dQc <= 0)                 # graph+centroid would not have moved v to B
    print(f"accepted GT-informative moves under active semantics: {a.sum()};  GT-improving {(gt[a]==1).mean()*100:.1f}%")
    print(f"   of which graph term alone opposed: {flip_graph.sum()} (GT-improving {(gt[flip_graph]==1).mean()*100:.1f}%)")
    print(f"   of which graph+centroid would have rejected: {cent_reject.sum()} (GT-improving {(gt[cent_reject]==1).mean()*100:.1f}%)")
    print(f"   scale check: mean |dQ_graph| {np.abs(dQg[active]).mean():.2e}  mean |dQ_desc| {np.abs(dQd[active]).mean():.2e}  mean |dQ_cent| {np.abs(dQc[active]).mean():.2e}")


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        report(ds)
