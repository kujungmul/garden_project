"""Partition state, objectives, Louvain-style local move and K-constrained merge.

All objectives share the graph-modularity bookkeeping (GraphState) and expose
    move_delta(v, A, B, kvA, kvB)   objective gain of moving v from A to B
    apply_move(v, A, B)
    merge_delta(A, B, eAB)          objective gain of merging B into A
    apply_merge(A, B)
    value()                         current objective value
"""
import numpy as np

# ----------------------------------------------------------------------------
# graph bookkeeping
# ----------------------------------------------------------------------------
class GraphState:
    def __init__(self, nbrs, deg, comm, K):
        self.nbrs, self.deg = nbrs, deg
        self.n = len(deg)
        self.m = int(deg.sum() // 2)
        self.comm = comm.astype(np.int64).copy()
        self.K = K                                   # number of columns (some may be dead)
        self.size = np.bincount(self.comm, minlength=K)
        self.dC = np.bincount(self.comm, weights=deg, minlength=K)
        self.mC = np.zeros(K)
        for v in range(self.n):
            cv = self.comm[v]
            self.mC[cv] += np.count_nonzero(self.comm[nbrs[v]] == cv)
        self.mC /= 2.0

    def alive(self):
        return np.flatnonzero(self.size > 0)

    def n_comm(self):
        return int((self.size > 0).sum())

    def q_terms(self):
        return self.mC / self.m - (self.dC / (2 * self.m)) ** 2

    def modularity(self):
        return float(self.q_terms().sum())

    def nbr_comm_counts(self, v):
        """{community: #neighbours of v in it}"""
        cnt = {}
        for u in self.nbrs[v]:
            c = self.comm[u]
            cnt[c] = cnt.get(c, 0) + 1
        return cnt

    def dq_move(self, v, A, B, kvA, kvB):
        m, dv = self.m, self.deg[v]
        return (kvB - kvA) / m + dv * (self.dC[A] - dv - self.dC[B]) / (2 * m * m)

    def apply_move(self, v, A, B, kvA, kvB):
        dv = self.deg[v]
        self.comm[v] = B
        self.size[A] -= 1; self.size[B] += 1
        self.dC[A] -= dv; self.dC[B] += dv
        self.mC[A] -= kvA; self.mC[B] += kvB

    def dq_merge(self, A, B, eAB):
        m = self.m
        return eAB / m - 2 * self.dC[A] * self.dC[B] / (4 * m * m)

    def apply_merge(self, A, B, eAB):
        self.comm[self.comm == B] = A
        self.size[A] += self.size[B]; self.size[B] = 0
        self.dC[A] += self.dC[B]; self.dC[B] = 0
        self.mC[A] += self.mC[B] + eAB; self.mC[B] = 0

    def inter_edges(self):
        """dict {(A,B): #edges} for A<B alive."""
        cnt = {}
        for v in range(self.n):
            cv = self.comm[v]
            for u in self.nbrs[v]:
                if u > v:
                    cu = self.comm[u]
                    if cu != cv:
                        key = (min(cu, cv), max(cu, cv))
                        cnt[key] = cnt.get(key, 0) + 1
        return cnt

    def relabel(self):
        """Compact labels to 0..K'-1; returns new label array."""
        alive = self.alive()
        remap = -np.ones(self.K, dtype=np.int64)
        remap[alive] = np.arange(len(alive))
        return remap[self.comm]


# ----------------------------------------------------------------------------
# objectives
# ----------------------------------------------------------------------------
class Modularity:
    """Plain graph modularity (lambda = 0)."""
    def __init__(self, gs):
        self.gs = gs

    def move_delta(self, v, A, B, kvA, kvB):
        return self.gs.dq_move(v, A, B, kvA, kvB)

    def apply_move(self, v, A, B, kvA, kvB):
        self.gs.apply_move(v, A, B, kvA, kvB)

    def merge_delta(self, A, B, eAB):
        return self.gs.dq_merge(A, B, eAB)

    def apply_merge(self, A, B, eAB):
        self.gs.apply_merge(A, B, eAB)

    def value(self):
        return self.gs.modularity()


class SNModularity:
    """Spatially-near modularity (Hannigan et al. 2013) with embedding centroids:
        sum_C  (m_C/m - (d_C/2m)^2) / (1 + mean_{i in C} ||e_i - x_C||^2 / sigma^2)
    Centroids are exact and updated online (sum/size)."""
    def __init__(self, gs, E, sigma):
        self.gs, self.E, self.s2 = gs, E, sigma ** 2
        K = gs.K
        self.S = np.zeros((K, E.shape[1]))
        np.add.at(self.S, gs.comm, E)
        self.Q = np.bincount(gs.comm, weights=(E * E).sum(1), minlength=K)

    def _term(self, qterm, S, Q, size):
        if size <= 0:
            return 0.0
        msd = max(Q / size - (S @ S) / (size * size), 0.0)
        return qterm / (1.0 + msd / self.s2)

    def _qterm(self, mC, dC):
        m = self.gs.m
        return mC / m - (dC / (2 * m)) ** 2

    def move_delta(self, v, A, B, kvA, kvB):
        gs, e = self.gs, self.E[v]
        ee = float(e @ e)
        dv = gs.deg[v]
        old = (self._term(self._qterm(gs.mC[A], gs.dC[A]), self.S[A], self.Q[A], gs.size[A])
               + self._term(self._qterm(gs.mC[B], gs.dC[B]), self.S[B], self.Q[B], gs.size[B]))
        new = (self._term(self._qterm(gs.mC[A] - kvA, gs.dC[A] - dv), self.S[A] - e, self.Q[A] - ee, gs.size[A] - 1)
               + self._term(self._qterm(gs.mC[B] + kvB, gs.dC[B] + dv), self.S[B] + e, self.Q[B] + ee, gs.size[B] + 1))
        return new - old

    def apply_move(self, v, A, B, kvA, kvB):
        e = self.E[v]; ee = float(e @ e)
        self.gs.apply_move(v, A, B, kvA, kvB)
        self.S[A] -= e; self.S[B] += e
        self.Q[A] -= ee; self.Q[B] += ee

    def merge_delta(self, A, B, eAB):
        gs = self.gs
        old = (self._term(self._qterm(gs.mC[A], gs.dC[A]), self.S[A], self.Q[A], gs.size[A])
               + self._term(self._qterm(gs.mC[B], gs.dC[B]), self.S[B], self.Q[B], gs.size[B]))
        new = self._term(self._qterm(gs.mC[A] + gs.mC[B] + eAB, gs.dC[A] + gs.dC[B]),
                         self.S[A] + self.S[B], self.Q[A] + self.Q[B], gs.size[A] + gs.size[B])
        return new - old

    def apply_merge(self, A, B, eAB):
        self.gs.apply_merge(A, B, eAB)
        self.S[A] += self.S[B]; self.S[B] = 0
        self.Q[A] += self.Q[B]; self.Q[B] = 0

    def value(self):
        gs = self.gs
        qt = gs.q_terms()
        return float(sum(self._term(qt[c], self.S[c], self.Q[c], gs.size[c]) for c in gs.alive()))


class RDM:
    """Representative Description Modularity with prototypes fixed during a phase:
        (1-lam) Q_graph + lam * (1/n) sum_v [ s(v, c(v)) - max_{j != c(v)} s(v, j) ]
    where s = (1 + cos(e_v, p_j)) / 2 and P (K x d) are prototype embeddings
    (LLM descriptions or centroids). Dead columns get -inf similarity."""
    def __init__(self, gs, E, P, lam, normalize=False):
        self.gs, self.E, self.lam = gs, E, lam
        self.n = gs.n
        self.scale = 1.0
        self.set_prototypes(P)
        if normalize:
            # Put the two terms on a comparable per-node scale: moving a node changes
            # Q_graph by about d_v/m (mean 2/n) and Q_sem by about (max_j s - mean_j s)/n.
            # scale = 2 / mean_v(max_j s_vj - mean_j s_vj), fixed for the whole run.
            S = self.Sim
            fin = np.isfinite(S)
            mx = S.max(1)
            mean = np.where(fin, S, 0).sum(1) / np.maximum(fin.sum(1), 1)
            self.scale = float(2.0 / max((mx - mean).mean(), 1e-6))

    def set_prototypes(self, P):
        gs = self.gs
        P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        S = (1.0 + self.E @ P.T) / 2.0
        S[:, gs.size == 0] = -np.inf
        self.Sim = S
        self._refresh_top()

    def retrieval_acc(self):
        """fraction of nodes whose most compatible prototype is their own community's"""
        return float(np.mean(self.top_idx[:, 0] == self.gs.comm))

    def reliability_test(self, margin=0.05, min_size=20, seed=0, exclude=None):
        """Semantic signal is deemed reliable when own-prototype retrieval accuracy clearly
        exceeds a permutation null (community labels shuffled among nodes, sizes kept).
        Only nodes in communities of size >= min_size that were not used to build the
        prototype are scored (small communities and fitted nodes inflate the accuracy).
        Returns (reliable, acc, null_acc)."""
        gs = self.gs
        mask = gs.size[gs.comm] >= min_size
        if exclude is not None:                 # nodes the prototypes were fitted on
            mask &= ~exclude
        if mask.sum() == 0:
            return False, 0.0, 0.0
        top = self.top_idx[mask, 0]
        acc = float(np.mean(top == gs.comm[mask]))
        perm = np.random.default_rng(seed).permutation(gs.comm[mask])
        null = float(np.mean(top == perm))
        return acc >= max(2 * null, null + margin), acc, null

    def _refresh_top(self):
        """top-3 columns per node over ALL alive columns (own included)."""
        S = self.Sim
        k = min(3, S.shape[1])
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(S, idx, 1)
        o = np.argsort(-vals, axis=1)
        self.top_idx = np.take_along_axis(idx, o, 1)
        self.top_val = np.take_along_axis(vals, o, 1)

    def kill_column(self, B):
        self.Sim[:, B] = -np.inf
        self._refresh_top()

    # --- per-node margins -------------------------------------------------
    def best_other(self, v, excl):
        """max_{j not in excl} Sim[v, j] using the top-3 cache."""
        for j, s in zip(self.top_idx[v], self.top_val[v]):
            if j not in excl:
                return s
        # fall back (rare: all top-3 excluded)
        row = self.Sim[v].copy()
        row[list(excl)] = -np.inf
        return row.max()

    def margin(self, v, c):
        return self.Sim[v, c] - self.best_other(v, (c,))

    def sem_value(self):
        c = self.gs.comm
        tot = 0.0
        own = self.Sim[np.arange(self.n), c]
        ti, tv = self.top_idx, self.top_val
        first_is_own = ti[:, 0] == c
        other = np.where(first_is_own, tv[:, 1], tv[:, 0])
        return float((own - other).mean())

    # --- move ------------------------------------------------------------
    def move_delta(self, v, A, B, kvA, kvB):
        dq = self.gs.dq_move(v, A, B, kvA, kvB)
        ds = self.margin(v, B) - self.margin(v, A)
        return (1 - self.lam) * dq + self.lam * self.scale * ds / self.n

    def apply_move(self, v, A, B, kvA, kvB):
        self.gs.apply_move(v, A, B, kvA, kvB)

    # --- merge -----------------------------------------------------------
    def _drop_gain_index(self):
        """For each column j: sum over nodes whose top competitor (excluding own) is j of
        (Sim[v,j] - next competitor). Used to credit nodes outside A,B when a column dies."""
        c = self.gs.comm
        ti, tv = self.top_idx, self.top_val
        n = self.n
        ar = np.arange(n)
        # competitor = first top entry != own ; next = following entry != own
        first_own = ti[:, 0] == c
        comp = np.where(first_own, ti[:, 1], ti[:, 0])
        compv = np.where(first_own, tv[:, 1], tv[:, 0])
        # next competitor
        second_own = ti[:, 1] == c
        nxt = np.where(first_own, tv[:, 2], np.where(second_own, tv[:, 2], tv[:, 1]))
        gain = compv - nxt
        gain[~np.isfinite(gain)] = 0.0
        self._comp, self._gain = comp, gain
        self._drop_gain = np.bincount(comp, weights=gain, minlength=self.gs.K)

    def merge_delta(self, A, B, eAB):
        """Merge B into A. The merged prototype is the better of {p_A, p_B}; the losing
        column dies. Exact change in Q_sem given the fixed prototype matrix."""
        gs = self.gs
        dq = gs.dq_merge(A, B, eAB)
        nodes = np.flatnonzero((gs.comm == A) | (gs.comm == B))
        cols = gs.comm[nodes]
        # old margins: own col minus best other (excluding own only)
        S = self.Sim[nodes]
        own_old = S[np.arange(len(nodes)), cols]
        S_old = S.copy(); S_old[np.arange(len(nodes)), cols] = -np.inf
        old = own_old - S_old.max(1)
        # new margins under prototype p in {A, B}: own = Sim[v,p], best other excludes both A,B
        S_new = S.copy(); S_new[:, [A, B]] = -np.inf
        other_new = S_new.max(1)
        best = None
        for p, dead in ((A, B), (B, A)):
            inside = float((S[:, p] - other_new - old).sum())
            # nodes outside A,B whose top competitor was `dead` gain (Sim[v,dead] - next)
            outside = self._drop_gain[dead] - float(self._gain[nodes][self._comp[nodes] == dead].sum())
            ds = inside + outside
            if best is None or ds > best[0]:
                best = (ds, p)
        ds, p = best
        return (1 - self.lam) * dq + self.lam * self.scale * ds / self.n, p

    def apply_merge(self, A, B, eAB, keep_proto):
        gs = self.gs
        before_idx, before_val = self.top_idx.copy(), self.top_val.copy()
        before_comp = self._comp.copy()
        if keep_proto == B:                      # keep B's prototype in column A
            self.Sim[:, A] = self.Sim[:, B]
        gs.apply_merge(A, B, eAB)
        self.kill_column(B)
        self._drop_gain_index()
        # Columns whose cached merge gains may have changed (audit fix, 2026-09-21).
        # merge_delta(X, Y) depends only on the top-3 rows of nodes in X u Y, on their own
        # community, and on _drop_gain[X], _drop_gain[Y] (sums over nodes by top competitor).
        # All of these are functions of (top_idx, top_val, comm, _comp), so a cached pair is
        # stale iff it touches the community or the old/new top competitor of a node whose
        # top-3 / own community / competitor changed in this merge.
        changed = ((self.top_idx != before_idx).any(1) | (self.top_val != before_val).any(1)
                   | (self._comp != before_comp) | (gs.comm == A))
        cols = set(gs.comm[changed].tolist()) | set(before_comp[changed].tolist()) | set(self._comp[changed].tolist())
        cols.update((int(A), int(B)))
        self._affected = cols

    def value(self):
        return (1 - self.lam) * self.gs.modularity() + self.lam * self.scale * self.sem_value()


class RDMAgree(RDM):
    """RDM with centroid prototypes plus a description 'advisor': a node may additionally be
    moved to a NON-adjacent community when its nearest centroid and its nearest description
    are the same community (agreement filter). The objective itself is unchanged."""
    def set_advisor(self, P_desc):
        P = P_desc / np.maximum(np.linalg.norm(P_desc, axis=1, keepdims=True), 1e-12)
        S = (1.0 + self.E @ P.T) / 2.0
        S[:, self.gs.size == 0] = -np.inf
        S[:, ~np.any(P_desc, axis=1)] = -np.inf            # communities without a description
        self._adv_top = S.argmax(1)

    mode = "agree"        # agree: nearest centroid == nearest description; cent: nearest centroid only (control);
                          # desc: nearest description only (control)

    def extra_candidate(self, v, A):
        jc = int(self.top_idx[v, 0])                         # nearest centroid (alive columns)
        jd = int(self._adv_top[v])                           # nearest description
        if self.mode == "agree":
            j = jc if jc == jd else None
        elif self.mode == "cent":
            j = jc
        else:
            j = jd if jd >= 0 and self.gs.size[jd] > 0 else None
        if j is not None and j != A and self.gs.size[j] > 0:
            return j
        return None


class RDM3:
    """Three-factor objective on one GraphState:
        (1 - wc - wd) Q_graph + wc * c_c * Qsem_cent + wd * c_d * Qsem_desc
    where the two semantic terms are contrastive margins under two independent prototype
    sets (centroids / LLM descriptions), each with its own normalisation scale and its own
    reliability gate. Implemented as two RDM objects sharing the graph bookkeeping."""
    def __init__(self, gs, E, P_cent, P_desc, wc, wd, normalize=True):
        self.gs = gs
        self.wg = 1.0 - wc - wd
        self.oc = RDM(gs, E, P_cent, wc, normalize=normalize)
        self.od = RDM(gs, E, P_desc, wd, normalize=normalize)
        self.wc0, self.wd0 = wc, wd

    def set_prototypes(self, P_cent, P_desc):
        self.oc.set_prototypes(P_cent); self.od.set_prototypes(P_desc)

    def _sem(self, o, ds):
        return o.lam * o.scale * ds / o.n

    def move_delta(self, v, A, B, kvA, kvB):
        dq = self.gs.dq_move(v, A, B, kvA, kvB)
        return (self.wg * dq + self._sem(self.oc, self.oc.margin(v, B) - self.oc.margin(v, A))
                + self._sem(self.od, self.od.margin(v, B) - self.od.margin(v, A)))

    def apply_move(self, v, A, B, kvA, kvB):
        self.gs.apply_move(v, A, B, kvA, kvB)

    def _drop_gain_index(self):
        self.oc._drop_gain_index(); self.od._drop_gain_index()

    def merge_delta(self, A, B, eAB):
        dq = self.gs.dq_merge(A, B, eAB)
        # each prototype set keeps its own better column; RDM.merge_delta returns the full
        # (1-lam)dq + lam*scale*ds/n, so strip the graph part and re-add it once
        tc, pc = self.oc.merge_delta(A, B, eAB)
        td, pd = self.od.merge_delta(A, B, eAB)
        sem_c = tc - (1 - self.oc.lam) * dq
        sem_d = td - (1 - self.od.lam) * dq
        return self.wg * dq + sem_c + sem_d, (pc, pd)

    def apply_merge(self, A, B, eAB, keep):
        pc, pd = keep
        gs = self.gs
        # replicate RDM.apply_merge for both matrices with a single graph update
        for o, kp in ((self.oc, pc), (self.od, pd)):
            o._before = (o.top_idx.copy(), o.top_val.copy(), o._comp.copy())
            if kp == B:
                o.Sim[:, A] = o.Sim[:, B]
        gs.apply_merge(A, B, eAB)
        aff = {int(A), int(B)}
        for o in (self.oc, self.od):
            bi, bv, bc = o._before
            o.kill_column(B); o._drop_gain_index()
            changed = ((o.top_idx != bi).any(1) | (o.top_val != bv).any(1) | (o._comp != bc) | (gs.comm == A))
            aff |= set(gs.comm[changed].tolist()) | set(bc[changed].tolist()) | set(o._comp[changed].tolist())
        self._affected = aff

    def value(self):
        return (self.wg * self.gs.modularity() + self.oc.lam * self.oc.scale * self.oc.sem_value()
                + self.od.lam * self.od.scale * self.od.sem_value())


# ----------------------------------------------------------------------------
# Louvain-style local move (no new communities; used on top of an initial partition)
# ----------------------------------------------------------------------------
def local_move(obj, gs, K_target, max_passes=20, rng=None, verbose=False):
    rng = rng or np.random.default_rng(0)
    n = gs.n
    total_moves = 0
    for p in range(max_passes):
        moves = 0
        for v in rng.permutation(n):
            A = gs.comm[v]
            cnt = gs.nbr_comm_counts(v)
            kvA = cnt.get(A, 0)
            if gs.size[A] == 1 and gs.n_comm() <= K_target:
                continue                        # never drop below K communities
            if hasattr(obj, "extra_candidate"):     # non-adjacent candidate proposed by the semantic advisor
                X = obj.extra_candidate(v, A)
                if X is not None:
                    cnt.setdefault(X, 0)
            best, bestB, bestk = 1e-12, A, 0
            for B, kvB in cnt.items():
                if B == A:
                    continue
                d = obj.move_delta(v, A, B, kvA, kvB)
                if d > best:
                    best, bestB, bestk = d, B, kvB
            if bestB != A:
                obj.apply_move(v, A, bestB, kvA, bestk)
                moves += 1
        total_moves += moves
        if verbose:
            print(f"    pass {p}: {moves} moves, obj={obj.value():.5f}, K={gs.n_comm()}")
        if moves == 0:
            break
    return total_moves


# ----------------------------------------------------------------------------
# greedy agglomerative merge until exactly K communities
# ----------------------------------------------------------------------------
def merge_to_k(obj, gs, K, verbose=False):
    inter = gs.inter_edges()
    if hasattr(obj, "_drop_gain_index"):
        obj._drop_gain_index()
    cache = {}                                   # (A,B) -> (delta, extra); recomputed when A or B changes
    while gs.n_comm() > K:
        pairs = dict(inter)
        # also allow merging the two "lightest" communities even if not adjacent
        # (e.g. small disconnected components); for modularity this is the best
        # non-adjacent candidate since its penalty is -2 d_A d_B / (2m)^2.
        alive = gs.alive()
        a, b = sorted(alive, key=lambda c: (gs.dC[c], gs.size[c]))[:2]
        key = (min(a, b), max(a, b))
        if key not in pairs:
            pairs[key] = 0
        best = None
        for (A, B), e in pairs.items():
            if (A, B) not in cache:
                r = obj.merge_delta(A, B, e)
                cache[(A, B)] = r if isinstance(r, tuple) else (r, None)
            d, extra = cache[(A, B)]
            if best is None or d > best[0]:
                best = (d, A, B, e, extra)
        d, A, B, e, extra = best
        if extra is None:
            obj.apply_merge(A, B, e)
        else:
            obj.apply_merge(A, B, e, extra)
        # update inter-community edge counts and invalidate cached pairs touching A or B
        new = {}
        for (X, Y), cnt in inter.items():
            if (X, Y) == (A, B):
                continue
            X2 = A if X == B else X
            Y2 = A if Y == B else Y
            k2 = (min(X2, Y2), max(X2, Y2))
            new[k2] = new.get(k2, 0) + cnt
        inter = new
        aff = getattr(obj, "_affected", None)      # RDM: exact-conservative set; others: only A, B
        if aff is None:
            aff = {A, B}
        cache = {k: v for k, v in cache.items() if k[0] not in aff and k[1] not in aff}
        if verbose and gs.n_comm() % 10 == 0:
            print(f"    merged -> K={gs.n_comm()} obj={obj.value():.5f}")
    return gs
