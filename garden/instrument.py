"""Move-level instrumentation (E): replay the GARDEN pipeline (LLM prototypes, cache hits for
seeds already run) and log, for every candidate node move evaluated by local_move,
    dQ_graph, d_margin_desc (what GARDEN uses), d_margin_cent (centroid prototypes of the SAME
    partition, frozen at the same phase start), the node's current margins under both prototype
    sets, source/destination communities, their majority GT classes, whether GARDEN accepted the
    move, and whether the centroid objective would have.
Decisions are unchanged (the logger only observes), so the trajectory is identical to run.py.
Output: results/moves_{dataset}_{seed}.npz
Usage: python instrument.py --dataset cora --seed 0
"""
import argparse, time
from pathlib import Path
import numpy as np

import core
from core import GraphState, RDM, merge_to_k
from data import load, node_embeddings, adjacency
import llm
from llm import Describer
llm.LEGACY_KEY = True        # replay of pre-audit runs: read the v2-keyed cache entries
from methods import louvain_init, _centroids
from run import DOMAIN

ROOT = Path(__file__).resolve().parent.parent / "results"


def local_move_logged(obj, cobj, gs, K_target, log, phase, maj, y, max_passes=20, rng=None):
    """core.local_move with a logging side channel. cobj = RDM with centroid prototypes on the same gs."""
    rng = rng or np.random.default_rng(0)
    n = gs.n
    total = 0
    for p in range(max_passes):
        moves = 0
        for v in rng.permutation(n):
            A = gs.comm[v]
            cnt = gs.nbr_comm_counts(v)
            kvA = cnt.get(A, 0)
            if gs.size[A] == 1 and gs.n_comm() <= K_target:
                continue
            best, bestB, bestk = 1e-12, A, 0
            mdA, mcA = obj.margin(v, A), cobj.margin(v, A)
            for B, kvB in cnt.items():
                if B == A:
                    continue
                d = obj.move_delta(v, A, B, kvA, kvB)
                dq = gs.dq_move(v, A, B, kvA, kvB)
                mdB, mcB = obj.margin(v, B), cobj.margin(v, B)
                log.append((phase, p, v, A, B, dq, mdB - mdA, mcB - mcA, mdA, mcA, mdB, mcB,
                            obj.lam, obj.scale, cobj.scale, y[v], maj[A], maj[B], gs.size[A], gs.size[B]))
                if d > best:
                    best, bestB, bestk = d, B, kvB
            if bestB != A:
                obj.apply_move(v, A, bestB, kvA, bestk)
                accepted.append((phase, p, v, A, bestB))
                moves += 1
        total += moves
        if moves == 0:
            break
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--max-iter", type=int, default=6)
    args = ap.parse_args()
    global accepted
    accepted = []
    ds = load(args.dataset); E = node_embeddings(ds); nbrs, deg = adjacency(ds); y = np.asarray(ds["y"])
    K = int(y.max() + 1)
    comm, K0 = louvain_init(ds, seed=args.seed)
    gs = GraphState(nbrs, deg, comm, K0)
    desc = Describer(ds, nbrs, E, "claude-haiku-4-5", DOMAIN[args.dataset], seed=args.seed, verbose=True)
    obj = cobj = None
    log = []
    prev = None
    t0 = time.time()
    for it in range(args.max_iter):
        alive = gs.alive()
        big = alive[gs.size[alive] >= 5]
        desc.update(gs.comm, big)
        P = desc.prototypes(gs.K)
        C = _centroids(E, gs.comm, gs.K)
        small = alive[gs.size[alive] < 5]
        if len(small):
            P[small] = C[small]
        if obj is None:
            obj = RDM(gs, E, P, args.lam, normalize=True)
            cobj = RDM(gs, E, C, args.lam, normalize=True)
        else:
            obj.set_prototypes(P); cobj.set_prototypes(C)
        excl = desc.fit_mask(gs.n)
        reliable, acc, chance = obj.reliability_test(exclude=excl)
        obj.lam = args.lam if reliable else 0.0
        maj = np.full(gs.K, -1)
        for c in alive:
            maj[c] = np.bincount(y[gs.comm == c], minlength=K).argmax()
        print(f"  phase {it}: K={gs.n_comm()} lam_eff={obj.lam} rel_acc={acc:.3f} null={chance:.3f} scale_desc={obj.scale:.2f} scale_cent={cobj.scale:.2f}", flush=True)
        local_move_logged(obj, cobj, gs, K, log, it, maj, y)
        if gs.n_comm() > K:
            merge_to_k(obj, gs, K)
            cobj.set_prototypes(_centroids(E, gs.comm, gs.K))   # keep cobj's column set consistent
        cur = gs.comm.copy()
        if prev is not None and np.array_equal(cur, prev):
            break
        prev = cur
    pred = gs.relabel()
    from evaluate import accuracy
    from sklearn.metrics import normalized_mutual_info_score as nmi
    ev = dict(NMI=round(float(nmi(y, pred)), 4), ACC=round(float(accuracy(y, pred)), 4), K=int(pred.max() + 1))
    L = np.array(log, dtype=np.float64)
    A = np.array(accepted, dtype=np.int64)
    np.savez_compressed(ROOT / f"moves_{args.dataset}_{args.seed}.npz", log=L, accepted=A, pred=pred, y=y,
                        cols="phase,pass,v,A,B,dq,dm_desc,dm_cent,mdA,mcA,mdB,mcB,lam_eff,scale_desc,scale_cent,y,majA,majB,sizeA,sizeB")
    print(f"logged {len(L)} candidate moves, {len(A)} accepted; {time.time()-t0:.0f}s; eval={ev}")


if __name__ == "__main__":
    main()
