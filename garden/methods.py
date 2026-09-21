"""Clustering pipelines. Every method: Louvain (graph only) -> refine under its objective
-> greedy merge to exactly K -> refine again, so that the comparison isolates the objective."""
import time

import numpy as np
import networkx as nx

from core import GraphState, Modularity, SNModularity, RDM, RDM3, RDMAgree, local_move, merge_to_k


def louvain_init(ds, seed=0, resolution=1.0):
    G = nx.Graph()
    G.add_nodes_from(range(ds["n"]))
    G.add_edges_from(map(tuple, ds["edges"]))
    comms = nx.community.louvain_communities(G, seed=seed, resolution=resolution)
    comm = np.zeros(ds["n"], dtype=np.int64)
    for i, c in enumerate(comms):
        comm[list(c)] = i
    return comm, len(comms)


def _centroids(E, comm, K):
    P = np.zeros((K, E.shape[1]), dtype=np.float32)
    np.add.at(P, comm, E)
    return P


def run_louvain_k(ds, nbrs, deg, K, init, verbose=False):
    comm, K0 = init
    gs = GraphState(nbrs, deg, comm, K0)
    obj = Modularity(gs)
    merge_to_k(obj, gs, K, verbose)
    local_move(obj, gs, K, verbose=verbose)
    return gs.relabel(), {"K0": K0}


def run_sn_centroid(ds, nbrs, deg, E, K, init, sigma_scale=1.0, verbose=False):
    comm, K0 = init
    e = ds["edges"]
    sigma = float(np.linalg.norm(E[e[:, 0]] - E[e[:, 1]], axis=1).mean()) * sigma_scale
    gs = GraphState(nbrs, deg, comm, K0)
    obj = SNModularity(gs, E, sigma)
    local_move(obj, gs, K, verbose=verbose)
    merge_to_k(obj, gs, K, verbose)
    local_move(obj, gs, K, verbose=verbose)
    return gs.relabel(), {"K0": K0, "sigma": sigma, "obj": obj.value()}


def run_rdm(ds, nbrs, deg, E, K, init, lam, proto="centroid", describer=None,
            max_iter=6, min_llm_size=5, reliability=True, normalize=False, verbose=False, adj_param=0.5):
    """proto in {centroid, llm}. Prototypes are frozen during each local-move /
    merge phase and regenerated between phases (lazy for llm)."""
    comm, K0 = init
    gs = GraphState(nbrs, deg, comm, K0)
    obj = None
    info = {"K0": K0, "iters": 0, "llm_calls": 0}
    prev = None
    for it in range(max_iter):
        alive = gs.alive()
        if proto == "centroid":
            P = _centroids(E, gs.comm, gs.K)
        elif proto in ("cagree", "cexp", "dexp"):     # centroid prototype + candidate expansion (agree / centroid-only / description-only)
            P = _centroids(E, gs.comm, gs.K)
            big = alive[gs.size[alive] >= min_llm_size]
            describer.update(gs.comm, big)
            Zadv = describer.prototypes(gs.K)
        elif proto in ("shift", "wmean", "wmargin", "trim", "selfw"):
            # description-ADJUSTED centroid (single prototype per community):
            #   shift : P_i = normalise( mu_i/|mu_i| + gamma * z_i )     (centroid pulled toward the description)
            #   wmean : P_i = sum_v w_v x_v,  w_v = softmax_v(tau * cos(x_v, z_i)) over members
            #           (members that fit the description weigh more; outliers of heterogeneous communities less)
            big = alive[gs.size[alive] >= min_llm_size]
            describer.update(gs.comm, big)
            Z = describer.prototypes(gs.K)
            C = _centroids(E, gs.comm, gs.K)
            C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
            P = C.copy()
            for c in big:
                z = Z[c]
                if not np.any(z):
                    continue
                mem = np.flatnonzero(gs.comm == c)
                if proto == "shift":
                    P[c] = C[c] + adj_param * z
                elif proto == "wmean":                       # softmax(tau * cos(x, z))
                    w = np.exp(adj_param * (E[mem] @ z)); w /= w.sum(); P[c] = w @ E[mem]
                elif proto == "wmargin":                     # softmax(tau * [cos(x, z_c) - max_{j!=c} cos(x, z_j)])
                    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
                    S = E[mem] @ Zn[big].T                   # members x described communities
                    own = S[:, list(big).index(c)]; S2 = S.copy(); S2[:, list(big).index(c)] = -np.inf
                    w = np.exp(adj_param * (own - S2.max(1))); w /= w.sum(); P[c] = w @ E[mem]
                elif proto == "trim":                        # plain mean of the top-(adj_param fraction) members by cos(x, z)
                    sc = E[mem] @ z; k = max(1, int(np.ceil(adj_param * len(mem))))
                    P[c] = E[mem[np.argsort(-sc)[:k]]].mean(0)
                elif proto == "selfw":                       # control: softmax(tau * cos(x, centroid)), no description
                    w = np.exp(adj_param * (E[mem] @ C[c])); w /= w.sum(); P[c] = w @ E[mem]
        else:
            # tiny communities (below min_llm_size) are not worth an LLM call: their
            # prototype is the member centroid (a singleton's own document).
            big = alive[gs.size[alive] >= min_llm_size]
            describer.update(gs.comm, big)
            P = describer.prototypes(gs.K)
            small = alive[gs.size[alive] < min_llm_size]
            if len(small):
                C = _centroids(E, gs.comm, gs.K)
                P[small] = C[small]
        if obj is None:
            obj = (RDMAgree if proto in ("cagree", "cexp", "dexp") else RDM)(gs, E, P, lam, normalize=normalize)
            info["sem_scale"] = obj.scale
            if proto in ("cagree", "cexp", "dexp"):
                obj.mode = {"cagree": "agree", "cexp": "cent", "dexp": "desc"}[proto]
        else:
            obj.set_prototypes(P)
        if proto in ("cagree", "cexp", "dexp"):
            obj.set_advisor(Zadv)
        excl = describer.fit_mask(gs.n) if describer is not None else None
        reliable, acc, chance = obj.reliability_test(exclude=excl)  # chance = permutation null
        obj.lam = lam if (reliable or not reliability) else 0.0
        info.setdefault("reliability", []).append(dict(acc=round(acc, 4), chance=round(chance, 4), lam_eff=obj.lam))
        if verbose:
            print(f"  iter {it}: K={gs.n_comm()} Q={gs.modularity():.4f} Qsem={obj.sem_value():.4f} "
                  f"retrieval_acc={acc:.3f} (chance {chance:.3f}) lam_eff={obj.lam} obj={obj.value():.4f}")
        moved = local_move(obj, gs, K, verbose=verbose)
        if gs.n_comm() > K:
            merge_to_k(obj, gs, K, verbose)
            moved += 1
        info["iters"] = it + 1
        cur = gs.comm.copy()
        if prev is not None and np.array_equal(cur, prev):
            break
        prev = cur
    info["Qsem"] = obj.sem_value()
    info["retrieval_acc"] = obj.retrieval_acc()
    pred = gs.relabel()
    if describer is not None:
        info["llm_calls"] = describer.calls
        txt = describer.texts()
        info["descriptions"] = {int(pred[np.flatnonzero(gs.comm == c)[0]]): txt[c] for c in gs.alive() if c in txt}
    return pred, info


def run_rdm3(ds, nbrs, deg, E, K, init, wc, wd, describer, max_iter=6, min_llm_size=5, reliability=True, verbose=False):
    """Graph + centroid-margin + description-margin objective (weights wc, wd; graph gets 1-wc-wd).
    Same phase structure as run_rdm: both prototype sets frozen within a phase, refreshed at phase start."""
    comm, K0 = init
    gs = GraphState(nbrs, deg, comm, K0)
    obj = None
    info = {"K0": K0, "iters": 0, "llm_calls": 0, "wc": wc, "wd": wd}
    prev = None
    for it in range(max_iter):
        alive = gs.alive()
        C = _centroids(E, gs.comm, gs.K)
        big = alive[gs.size[alive] >= min_llm_size]
        describer.update(gs.comm, big)
        P = describer.prototypes(gs.K)
        small = alive[gs.size[alive] < min_llm_size]
        if len(small):
            P[small] = C[small]
        if obj is None:
            obj = RDM3(gs, E, C, P, wc, wd, normalize=True)
            info["sem_scale"] = [obj.oc.scale, obj.od.scale]
        else:
            obj.set_prototypes(C, P)
        excl = describer.fit_mask(gs.n)
        rc = obj.oc.reliability_test(); rd = obj.od.reliability_test(exclude=excl)
        obj.oc.lam = wc if (rc[0] or not reliability) else 0.0
        obj.od.lam = wd if (rd[0] or not reliability) else 0.0
        info.setdefault("reliability", []).append(dict(acc_cent=round(rc[1], 4), acc_desc=round(rd[1], 4), chance=round(rd[2], 4), wc_eff=obj.oc.lam, wd_eff=obj.od.lam))
        if verbose:
            print(f"  iter {it}: K={gs.n_comm()} Q={gs.modularity():.4f} obj={obj.value():.4f} wc_eff={obj.oc.lam} wd_eff={obj.od.lam}")
        local_move(obj, gs, K, verbose=verbose)
        if gs.n_comm() > K:
            merge_to_k(obj, gs, K, verbose)
        info["iters"] = it + 1
        cur = gs.comm.copy()
        if prev is not None and np.array_equal(cur, prev):
            break
        prev = cur
    info["Qsem"] = obj.od.sem_value(); info["Qsem_cent"] = obj.oc.sem_value()
    info["retrieval_acc"] = obj.od.retrieval_acc()
    pred = gs.relabel()
    info["llm_calls"] = describer.calls
    txt = describer.texts()
    info["descriptions"] = {int(pred[np.flatnonzero(gs.comm == c)[0]]): txt[c] for c in gs.alive() if c in txt}
    return pred, info
