import itertools
import math

import numpy as np
from scipy.stats import t as _t


FLOOR_ROWS = 0.5


def bench_effects(x, rows, benches=None, floor_rows=None):
    floor_rows = FLOOR_ROWS if floor_rows is None else floor_rows
    byb = {}
    for r, v in x.items():
        b = rows[r]["bench"]
        if benches is not None and b not in benches:
            continue
        byb.setdefault(b, {}).setdefault(rows[r]["pic"], []).append(float(v))
    out = {}
    for b, cl in byb.items():
        n = sum(len(v) for v in cl.values())
        G = len(cl)
        d = sum(sum(v) for v in cl.values()) / n
        ss = sum((sum(v) - d * len(v)) ** 2 for v in cl.values())
        se = math.sqrt(ss * G / max(G - 1, 1)) / n
        out[b] = dict(d=d, se=max(se, floor_rows / n), se_raw=se, n=n, G=G)
    return out


def macro(E, S):
    S = [b for b in S if b in E]
    k = len(S)
    if k == 0:
        return dict(d=float("nan"), se=float("nan"), nu=1.0, k=0)
    d = sum(E[b]["d"] for b in S) / k
    se = math.sqrt(sum(E[b]["se"] ** 2 for b in S)) / k
    return dict(d=d, se=se, nu=float(max(sum(E[b]["G"] - 1 for b in S), 1)), k=k)


def p_macro(E, S):
    m = macro(E, S)
    if m["k"] == 0:
        return 1.0
    return float(_t.sf(m["d"] / m["se"], m["nu"]))


def p_gen(E, S, min_k=3):
    S = [b for b in S if b in E]
    if len(S) < min_k:
        return 1.0, None
    worst, who = -1.0, None
    for b in S:
        p = p_macro(E, [c for c in S if c != b])
        if p > worst:
            worst, who = p, b
    return worst, who


def lobo_point(E, S):
    S = [b for b in S if b in E]
    if len(S) < 2:
        return float("-inf")
    return min(macro(E, [c for c in S if c != b])["d"] for b in S)


def p_bench_signflip(E, S):
    d = np.array([E[b]["d"] for b in S if b in E], float)
    k = len(d)
    if k == 0:
        return 1.0
    obs = d.sum()
    cnt = 0
    for s in itertools.product((-1.0, 1.0), repeat=k):
        cnt += float(np.dot(s, d)) >= obs - 1e-12
    return cnt / 2 ** k


def p_bench_t(E, S):
    d = np.array([E[b]["d"] for b in S if b in E], float)
    k = len(d)
    if k < 2:
        return 1.0
    sd = d.std(ddof=1)
    if sd <= 0:
        return 0.0 if d.mean() > 0 else 1.0
    return float(_t.sf(d.mean() / (sd / math.sqrt(k)), k - 1))


def holm(pvals, alpha=0.10):
    p = np.asarray(pvals, float)
    K = len(p)
    acc = np.zeros(K, bool)
    for j, i in enumerate(np.argsort(p, kind="mergesort")):
        if p[i] <= alpha / (K - j):
            acc[i] = True
        else:
            break
    return acc


class Design:

    def __init__(self, row_ids, rows, benches=None):
        self.rows = list(row_ids)
        self.benches = list(benches) if benches is not None else sorted({rows[r]["bench"] for r in self.rows})
        bi = {b: i for i, b in enumerate(self.benches)}
        self.b = np.array([bi[rows[r]["bench"]] for r in self.rows])
        cl = {}
        self.c = np.array([cl.setdefault((rows[r]["bench"], rows[r]["pic"]), len(cl)) for r in self.rows])
        self.nc = len(cl)
        self.cb = np.zeros(self.nc, int)
        self.cb[self.c] = self.b
        self.n_b = np.bincount(self.b, minlength=len(self.benches)).astype(float)
        self.M_rc = np.zeros((len(self.rows), self.nc))
        self.M_rc[np.arange(len(self.rows)), self.c] = 1.0
        self.M_cb = np.zeros((self.nc, len(self.benches)))
        self.M_cb[np.arange(self.nc), self.cb] = 1.0

    def _flips(self, n, seed):
        key = (n, seed)
        if getattr(self, "_F", None) is None or self._F[0] != key:
            rng = np.random.default_rng(seed)
            self._F = (key, (rng.integers(0, 2, size=(n, self.nc), dtype=np.int8) * 2 - 1).astype(np.float32))
        return self._F[1]

    def signflip(self, x, S=None, n=20000, seed=0, min_k=3):
        x = np.asarray(x, np.float32)
        if not np.isfinite(x).all():
            raise ValueError("signflip: the row differences are not all finite")
        idx = np.arange(len(self.benches)) if S is None else np.array([self.benches.index(b) for b in S])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            cs = x @ self.M_rc.astype(np.float32)
            F = self._flips(n, seed)
            Mcb = (self.M_cb / self.n_b).astype(np.float32)
            db = ((F * cs) @ Mcb)[:, idx]
            ob = (cs @ Mcb)[idx]
        if not (np.isfinite(db).all() and np.isfinite(ob).all()):
            raise ValueError("signflip: the sign-flip sums are not finite")
        k = len(idx)
        tol = 1e-7
        pm = (1 + int((db.sum(1) >= ob.sum() - tol).sum())) / (n + 1)
        p2 = min(1.0, (1 + int((np.abs(db.sum(1)) >= abs(ob.sum()) - tol).sum())) / (n + 1))
        pl = []
        for j in range(k):
            keep = [i for i in range(k) if i != j]
            pl.append((1 + int((db[:, keep].sum(1) >= ob[keep].sum() - tol).sum())) / (n + 1) if keep else 1.0)
        pneg = [(1 + int((db[:, j] <= ob[j] + tol).sum())) / (n + 1) for j in range(k)]
        j = int(np.argmax(pl)) if pl else None
        return dict(p_macro=float(pm), p_two_sided=float(p2), p_gen=float(max(pl)) if k >= min_k else 1.0, carrier=j, p_lobo=[float(v) for v in pl], p_neg=[float(v) for v in pneg])
