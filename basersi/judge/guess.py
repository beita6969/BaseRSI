import collections
import re

import numpy as np

from basersi import config
from basersi.data.rows import SIGNAL_ROLES, RowsError

SEED = "guess"
MIN_PER_BENCH = 3
MIN_READS = 15
MIN_POS = 4
KEEP_P = 0.05
MAX_ADD = 8
_WORD = re.compile(r"[a-z']+")


def words(tail, window):
    return _WORD.findall((tail or "")[-window:].lower())


def grams(ws, nmax=3):
    out = set()
    for n in range(1, nmax + 1):
        for i in range(len(ws) - n + 1):
            out.add(" ".join(ws[i:i + n]))
    return out


def flag(det, tail):
    t = " " + " ".join(words(tail, det["window"])) + " "
    if (" " + det["seed"]) in t:
        return True
    return any((" " + p + " ") in t for p in det["patterns"])


class _Data:
    def __init__(self, reads, rows, window, min_benchmarks):
        for x in reads:
            if rows[x["row"]]["role"] not in SIGNAL_ROLES:
                raise RowsError(f"guess detector: row {x['row']} is not a show / monitor row")
        self.benches = sorted({rows[x["row"]]["bench"] for x in reads})
        bi = {b: i for i, b in enumerate(self.benches)}
        byrow = collections.defaultdict(list)
        for i, x in enumerate(reads):
            byrow[x["row"]].append(i)
        n = len(reads)
        self.ok = np.array([float(x["ok"]) for x in reads])
        self.b = np.array([bi[rows[x["row"]]["bench"]] for x in reads])
        self.c = np.zeros(n)
        self.t = np.zeros(n)
        self.w = np.zeros(n)
        tok = np.array([float(x["tokens"]) for x in reads])
        for r, ix in byrow.items():
            if len(ix) < 2:
                continue
            so, st = self.ok[ix].sum(), tok[ix].sum()
            for i in ix:
                self.c[i] = (so - self.ok[i]) / (len(ix) - 1) - self.ok[i]
                self.t[i] = (st - tok[i]) / (len(ix) - 1)
                self.w[i] = 1.0 / len(ix)
        self.rows_b = np.array([sum(1 for r in byrow if rows[r]["bench"] == b) for b in self.benches], float)
        self.base_tok_b = np.array([sum(tok[ix].mean() for r, ix in byrow.items() if rows[r]["bench"] == b) for b in self.benches]) / self.rows_b
        self.texts = [" " + " ".join(words(x["tail"], window)) + " " for x in reads]
        self.seed = np.array([(" " + SEED) in t for t in self.texts])
        df = collections.Counter()
        dfb = collections.defaultdict(collections.Counter)
        self.gr = []
        for i, x in enumerate(reads):
            g = grams(words(x["tail"], window))
            self.gr.append(g)
            df.update(g)
            for p in g:
                dfb[p][self.b[i]] += 1
        self.cands = sorted(p for p, k in df.items() if k >= MIN_READS and sum(v >= MIN_PER_BENCH for v in dfb[p].values()) >= min_benchmarks
                            and SEED not in p)
        self.M = {p: np.array([p in g for g in self.gr]) for p in self.cands}

    def gain_cost(self, fl):
        B = len(self.benches)
        G = np.bincount(self.b, weights=fl * self.c * self.w, minlength=B) / self.rows_b
        C = np.bincount(self.b, weights=fl * self.t * self.w, minlength=B) / self.rows_b / self.base_tok_b
        return G, C

    def value(self, fl, price, benches=None):
        G, C = self.gain_cost(fl)
        v = G - price * C
        if benches is not None:
            return v[benches]
        return v


def select(D, price, train):
    fl = D.seed.copy()
    pats = []
    for _ in range(MAX_ADD):
        v0 = D.value(fl, price, train)
        best, bestgain, bestfl = None, 0.0, None
        for p in D.cands:
            if p in pats:
                continue
            f2 = fl | D.M[p]
            if f2.sum() == fl.sum():
                continue
            v = D.value(f2, price, train)
            if (v > v0 + 1e-12).sum() >= min(MIN_POS, len(train)) and v.mean() - v0.mean() > bestgain + 1e-12:
                best, bestgain, bestfl = p, v.mean() - v0.mean(), f2
        if best is None:
            break
        pats.append(best)
        fl = bestfl
    return pats, fl


def mean_or_none(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def min_benchmarks_default():
    return int(config.load("judge")["guess_min_benchmarks"])


def derive(reads, rows, price, windows=None, min_benchmarks=None):
    if min_benchmarks is None:
        min_benchmarks = min_benchmarks_default()
    out = dict(seed=SEED, price=price, windows={})
    best = None
    for W in (windows or tuple(config.load("judge")["guess_detector_search_windows"])):
        D = _Data(reads, rows, W, min_benchmarks)
        B = len(D.benches)
        lobo_seed, lobo_sel, picked = np.zeros(B), np.zeros(B), {}
        gs, cs, gl, cl = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
        for h in range(B):
            train = [i for i in range(B) if i != h]
            pats, fl = select(D, price, train)
            picked[D.benches[h]] = pats
            lobo_seed[h] = D.value(D.seed, price)[h]
            lobo_sel[h] = D.value(fl, price)[h]
            G, C = D.gain_cost(D.seed)
            gs[h], cs[h] = G[h], C[h]
            G, C = D.gain_cost(fl)
            gl[h], cl[h] = G[h], C[h]
        pats, fl = select(D, price, list(range(B)))
        G, C = D.gain_cost(fl)
        rec = dict(window=W, patterns=pats, candidates=len(D.cands),
                   seed=dict(share=float(D.seed.mean()), acc_flagged=mean_or_none(D.ok, D.seed),
                             acc_unflagged=mean_or_none(D.ok, ~D.seed), G=float(gs.mean()), C=float(cs.mean()), V=float(lobo_seed.mean()),
                             V_b={b: float(v) for b, v in zip(D.benches, lobo_seed)}, G_b={b: float(v) for b, v in zip(D.benches, gs)}),
                   lobo=dict(G=float(gl.mean()), C=float(cl.mean()), V=float(lobo_sel.mean()), V_b={b: float(v) for b, v in zip(D.benches, lobo_sel)},
                             G_b={b: float(v) for b, v in zip(D.benches, gl)}, picked=picked, better_in=int((lobo_sel > lobo_seed + 1e-12).sum())),
                   final=dict(share=float(fl.mean()), acc_flagged=mean_or_none(D.ok, fl), acc_unflagged=mean_or_none(D.ok, ~fl),
                              G=float(G.mean()), C=float(C.mean()), V=float((G - price * C).mean())))
        out["windows"][W] = rec
        diff = lobo_sel - lobo_seed
        signs = np.array(list(__import__("itertools").product((-1.0, 1.0), repeat=B)))
        rec["lobo"]["p_signflip"] = float(((signs @ diff) >= diff.sum() - 1e-15).mean())
        for use_sel, v in ((False, float(lobo_seed.mean())), (True, float(lobo_sel.mean()))):
            if use_sel and (rec["lobo"]["p_signflip"] > KEEP_P or diff.sum() <= 0):
                continue
            if best is None or v > best[0] + 1e-12:
                best = (v, W, pats if use_sel else [], dict(picked) if use_sel else {})
    out.update(window=best[1], patterns=best[2], lobo_value=best[0], picked=best[3])
    return out
