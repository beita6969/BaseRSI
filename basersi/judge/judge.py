import json
import math

import numpy as np

from basersi import config
from basersi.judge import frontier as P
from basersi.judge import stats as ST

def tail_chars():
    return int(config.load("loop")["judge_export"]["tail_chars"])


def settings(cfg):
    tail = tail_chars()
    for key in ("guess_detector_search_windows", "guess_policy_windows_entry_audit", "guess_policy_windows_weight_edits"):
        over = [int(w) for w in cfg[key] if int(w) > tail]
        if over:
            raise config.ConfigError(f"judge config {key}: windows {over} are longer than the {tail} tail characters every export keeps (loop.yaml judge_export.tail_chars)")
    return dict(alpha=float(cfg["alpha"]), alpha_neg=float(cfg["alpha_negative_benchmark"]), min_scored=int(cfg["min_scored_benchmarks"]),
                nflip=int(cfg["sign_flip"]["count"]), seed=int(cfg["sign_flip"]["seed"]))


def judge_config(override=None):
    return config.load("judge", override)


def settings_of_pipeline(pipeline_cfg):
    return settings(judge_config(((pipeline_cfg or {}).get("judge") or {}).get("config_override")))


SHIPPED = settings(judge_config())
ABOVE_KEYS = ("above_frontier", "candidate_tokens", "frontier_max_tokens", "frontier_max_multiple")


class JudgeError(config.Refused, ValueError):
    pass


def arm_values(arm, row_ids):
    v, t, T, n = {}, {}, {}, {}
    for r in row_ids:
        xs = arm.get(r)
        if xs:
            v[r] = sum(x["ok"] for x in xs) / len(xs)
            t[r] = sum(x["tokens"] for x in xs) / len(xs)
            T[r] = sum(x["tokens"] + float(x.get("prompt_tokens") or 0.0) for x in xs) / len(xs)
            n[r] = len(xs)
    return v, t, T, n


class Judge:
    def __init__(self, rows, tip_arm, benches=None, strict=False, hull_from=None, policies=None, rng=None,
                 frontier=None, roles=("judge",), hull_from_total=None, settings=None):
        self.s = dict(settings or SHIPPED)
        self.rows = rows
        self.roles = tuple(roles)
        if "judge" in self.roles and len(self.roles) > 1:
            raise JudgeError("JUDGE rows are never pooled with SHOW / MONITOR rows")
        self.J = sorted(r for r, x in rows.items() if x["role"] in self.roles)
        miss = [r for r in self.J if not tip_arm.get(r)]
        if miss:
            raise JudgeError(f"the tip has no cached reads on {len(miss)} of the {'/'.join(self.roles)} rows")
        self.benches = list(benches) if benches is not None else sorted({rows[r]["bench"] for r in self.J})
        self.alpha = self.s["alpha"]
        self.alpha_neg = self.s["alpha_neg"]
        self.strict = strict
        self.tip = {r: tip_arm[r] for r in self.J}
        self.F = frontier if frontier is not None else P.Frontier(self.tip, self.J, rows, policies=policies, rng=rng, hull_from=hull_from)
        self.Ft = P.Frontier(self.tip, self.J, rows, policies=list(self.F.pols.values()), cost="total", table=self.F.table, hull_from=hull_from_total)
        self.tip_v = {r: self.F.table["vote1"][r]["ok"] for r in self.J}
        self.tip_t = {r: self.F.table["vote1"][r]["tokens"] for r in self.J}
        self.tip_T = {r: self.F.table["vote1"][r]["total"] for r in self.J}
        self.base_tokens = sum(self.tip_t.values()) / len(self.J)
        self.base_total = sum(self.tip_T.values()) / len(self.J)

    def _check_rows(self, d, what):
        bad = [r for r in d if r not in self.tip]
        if bad:
            raise JudgeError(f"{what}: {len(bad)} rows are not {'/'.join(self.roles)} rows (e.g. {bad[0]})")

    def cand_from_arm(self, name, arm, kind, source=(), routed=None, **kw):
        rows = self.J if routed is None else [r for r in routed if r in self.tip]
        v, t, T, n = arm_values(arm, rows)
        miss = [r for r in rows if r not in v]
        if miss:
            raise JudgeError(f"{name}: no reads on {len(miss)} of its {len(rows)} JUDGE rows")
        return dict(name=name, kind=kind, source=list(source), value=v, tokens=t, total=T, samples=n, routed=None if routed is None else sorted(rows), **kw)

    def system_of_arm(self, arm, routed=None):
        rows = self.J if routed is None else [r for r in routed if r in self.tip]
        v, t, T, _ = arm_values(arm, rows)
        return dict(value=v, tokens=t, total=T)

    def cand_from_policy(self, pol, extra_arm=None, kind="policy", source=(), **kw):
        tb = P.evaluate(pol, self.tip, self.J, extra_arm=extra_arm)
        return dict(name=kw.pop("name", P.name_of(pol)), kind=kind, source=list(source), value={r: x["ok"] for r, x in tb.items()},
                    tokens={r: x["tokens"] for r, x in tb.items()}, total={r: x["total"] for r, x in tb.items()},
                    esc=sum(x["esc"] for x in tb.values()) / len(tb), policy=pol, routed=None, **kw)

    def cand_from_views(self, name, pol, arms, kind="policy", source=(), **kw):
        A = {"tip": self.tip}
        A.update({v: {r: a[r] for r in self.J if r in a} for v, a in arms.items()})
        tb = P.evaluate_views(pol, A, self.J)
        return dict(name=name, kind=kind, source=list(source), value={r: x["ok"] for r, x in tb.items()}, tokens={r: x["tokens"] for r, x in tb.items()},
                    total={r: x["total"] for r, x in tb.items()}, esc=sum(x["esc"] for x in tb.values()) / len(tb), policy=pol, routed=None, **kw)

    def system(self, c, cost="tokens"):
        self._check_rows(c["value"], c["name"])
        tip_c = self.tip_t if cost == "tokens" else self.tip_T
        v = {r: c["value"].get(r, self.tip_v[r]) for r in self.J}
        cc = c.get(cost) or {}
        if cost == "total" and not cc:
            return v, None
        t = {r: cc.get(r, tip_c[r]) for r in self.J}
        return v, t

    def _ref(self, m, cost="tokens"):
        F = self.F if cost == "tokens" else self.Ft
        if m <= 1.0 + 1e-9:
            return dict(self.tip_v), dict(lo="vote1", hi="vote1", w=0.0, multiple=1.0, strict=self.strict, tokens=F.base_tokens, acc=F.vertices[0][1])
        ref = F.ref_at(m * F.base_tokens, strict=self.strict)
        if ref is None:
            return None, None
        return ref[0], ref[2]

    def above(self, m, cost="tokens"):
        F = self.F if cost == "tokens" else self.Ft
        top = max(p[0] for p in F.points)
        word = "generated" if cost == "tokens" else "total"
        return dict(multiple=m, above_frontier=True, candidate_tokens=m * F.base_tokens, frontier_max_tokens=top, frontier_max_multiple=top / F.base_tokens,
                    error=f"the candidate uses {m * F.base_tokens:.0f} {word} tokens per row (x{m:.2f} of one reference read), more than the reference"
                          f" frontier covers ({top:.0f}, x{top / F.base_tokens:.2f}); no comparison at equal compute exists")

    def diffs(self, c, cost="tokens"):
        v, t = self.system(c, cost)
        if t is None:
            return None, dict(multiple=None, error="no total tokens for this candidate")
        base_cost = self.base_tokens if cost == "tokens" else self.base_total
        tip_c = self.tip_t if cost == "tokens" else self.tip_T
        Tc = sum(t.values()) / len(self.J)
        if c.get("base") is not None:
            bv = {r: c["base"]["value"].get(r, self.tip_v[r]) for r in self.J}
            bc = c["base"].get(cost) or {}
            bt = {r: bc.get(r, tip_c[r]) for r in self.J}
            Tb = sum(bt.values()) / len(self.J)
            m = Tc / Tb
            ref, desc = self._ref(m, cost)
            if ref is None:
                return None, self.above(m, cost)
            x = {r: v[r] - bv[r] - (ref[r] - self.tip_v[r]) for r in self.J}
            raw = {r: v[r] - bv[r] for r in self.J}
        else:
            m = Tc / base_cost
            ref, desc = self._ref(m, cost)
            if ref is None:
                return None, self.above(m, cost)
            x = {r: v[r] - ref[r] for r in self.J}
            raw = {r: v[r] - self.tip_v[r] for r in self.J}
        return x, dict(multiple=m, ref=desc, raw=raw, tokens=Tc)

    def _block(self, x, S):
        if getattr(self, "_D", None) is None:
            self._D = ST.Design(self.J, self.rows, self.benches)
        E = ST.bench_effects(x, self.rows, floor_rows=0.0)
        Et = ST.bench_effects(x, self.rows)
        m = ST.macro(E, S)
        sf = self._D.signflip([x[r] for r in self.J], S, n=self.s["nflip"], seed=self.s["seed"], min_k=self.s["min_scored"])
        pgt, _ = ST.p_gen(Et, S, min_k=self.s["min_scored"])
        neg = [(b, p) for b, p in zip(S, sf["p_neg"]) if p < self.alpha_neg / len(S)]
        return dict(d=m["d"], se=m["se"], p_macro=sf["p_macro"], p_gen=sf["p_gen"], carrier=(S[sf["carrier"]] if sf["carrier"] is not None and len(S) >= self.s["min_scored"] else None),
                    p_macro_t=ST.p_macro(Et, S), p_gen_t=pgt, lobo_min=ST.lobo_point(E, S),
                    p_bench_signflip=ST.p_bench_signflip(E, S), p_bench_t=ST.p_bench_t(E, S), neg=neg,
                    npos=sum(E[b]["d"] > 0 for b in S), k=len(S), per_bench={b: dict(d=E[b]["d"], se=E[b]["se_raw"], n=E[b]["n"]) for b in self.benches})

    def p_two_sided(self, x, S):
        if getattr(self, "_D", None) is None:
            self._D = ST.Design(self.J, self.rows, self.benches)
        return self._D.signflip([x[r] for r in self.J], S, n=self.s["nflip"], seed=self.s["seed"])["p_two_sided"]

    def score(self, c):
        S = [b for b in self.benches if b not in set(c.get("source") or ())]
        x, info = self.diffs(c)
        rec = dict(name=c["name"], kind=c["kind"], source=list(c.get("source") or ()), scored=S, multiple=info.get("multiple"),
                   routed_rows=None if c.get("routed") is None else len(c["routed"]), esc=c.get("esc"), policy=c.get("policy"), note=c.get("note"))
        if x is None:
            rec.update(error=info["error"], judged=False, **{k: info[k] for k in ABOVE_KEYS if k in info})
            return rec
        rec.update(judged=True, ref=info["ref"], tokens=info["tokens"], eq=self._block(x, S), raw=self._block(info["raw"], S),
                   pooled=self._block(x, self.benches))
        rec["acc"] = sum(self.system(c)[0].values()) / len(self.J)
        xt, tinfo = self.diffs(c, "total")
        if xt is not None:
            bt = self._block(xt, S)
            rec["total"] = dict(multiple=tinfo["multiple"], ref=tinfo["ref"], tokens=tinfo["tokens"], d=bt["d"], se=bt["se"], p_macro=bt["p_macro"],
                                p_gen=bt["p_gen"], per_bench=bt["per_bench"])
        else:
            rec["total"] = dict(multiple=tinfo.get("multiple"), error=tinfo.get("error"))
        return rec

    def source_effect(self, c, benches, rows=None):
        x, info = self.diffs(c)
        if x is None:
            return dict(judged=False, error=info.get("error"))
        rs = [r for r in self.J if self.rows[r]["bench"] in set(benches) and (rows is None or r in rows)]
        if not rs:
            return dict(judged=False, error="no row of the source benchmark")
        xs = {r: x[r] for r in rs}
        E = ST.bench_effects(xs, self.rows, floor_rows=0.0)
        bs = sorted(E)
        m = ST.macro(E, bs)
        sf = ST.Design(rs, self.rows, bs).signflip([xs[r] for r in rs], n=self.s["nflip"], seed=self.s["seed"])
        raw = sum(info["raw"][r] for r in rs) / len(rs)
        return dict(judged=True, d=m["d"], se=m["se"], p=sf["p_macro"], raw=raw, rows=len(rs), routed=sum(1 for r in rs if r in (c.get("routed") or rs)),
                    multiple=info["multiple"], benches=bs)

    def decide(self, cands, family="generation"):
        recs = [self.score(c) for c in cands]
        live = [r for r in recs if r.get("judged") and r["eq"]["k"] >= self.s["min_scored"]]
        for r in live:
            r["p"] = r["eq"]["p_gen"]
        acc = ST.holm([r["p"] for r in live], self.alpha) if live else []
        order = np.argsort([r["p"] for r in live], kind="mergesort") if live else []
        K = len(live)
        thr = {}
        for j, i in enumerate(order):
            thr[live[i]["name"]] = self.alpha / (K - j)
        for r in recs:
            r["family"], r["K"] = family, K
            if not r.get("judged"):
                r.update(accept=False, reason=r.get("error", "not judged"))
                continue
            if r["eq"]["k"] < self.s["min_scored"]:
                r.update(accept=False, reason=f"only {r['eq']['k']} scored benchmarks (< {self.s['min_scored']})")
                continue
        for r, a in zip(live, acc):
            g = r["eq"]
            r["holm_threshold"] = thr[r["name"]]
            if not a:
                why = ((f"p_gen {r['p']:.4f} > Holm {thr[r['name']]:.4f} (K {K})" if r["p"] > thr[r["name"]] else
                        f"p_gen {r['p']:.4f} <= its Holm threshold {thr[r['name']]:.4f} (K {K}), but the Holm step-down stopped at a smaller p")
                       + (f"; macro {100 * g['d']:+.2f} pp at equal compute" if g["p_macro"] > self.alpha else f"; carried by {g['carrier']}"))
                r.update(accept=False, reason=why)
            elif g["neg"]:
                r.update(accept=False, reason="significantly negative on " + ", ".join(b for b, _ in g["neg"]))
            else:
                r.update(accept=True, reason=f"p_gen {r['p']:.4f} <= Holm {thr[r['name']]:.4f} (K {K}), no negative benchmark")
        return recs

    def audit_entries(self, cands):
        recs = self.decide(cands, family="entry audit")
        live = [r for r in recs if r.get("judged")]
        pooled = ST.holm([r["pooled"]["p_macro"] for r in live], self.alpha) if live else []
        for r, a in zip(live, pooled):
            if r["accept"]:
                r["layer"] = "GENERAL"
            elif a:
                r["layer"] = "ID_SPECIFIC"
                r["reason"] += f" | pooled over all benchmarks {100 * r['pooled']['d']:+.2f} pp (p {r['pooled']['p_macro']:.4f}): in-distribution only -> specialisation layer"
            else:
                r["layer"] = "UNPROVEN"
                r["reason"] += f" | pooled over all benchmarks {100 * r['pooled']['d']:+.2f} pp (p {r['pooled']['p_macro']:.4f}, not significant in the audit family): removal costs nothing measurable"
        for r in recs:
            if "layer" not in r:
                r["layer"] = "NOT_JUDGED"
                r["reason"] = (f"the audit could not judge this entry ({r.get('error') or r.get('reason') or 'no comparison'}); it stays in the specialisation layer,"
                               " where it still faces the admission test on its source benchmark; it is never general")
        return recs


def short(b):
    return b[:5]


def ref_text(d):
    if not d:
        return "-"
    if d["lo"] == d["hi"] or d["w"] in (0.0, 1.0):
        return (f"{d['hi'] if d['w'] == 1.0 else d['lo']} (x{d['multiple']:.2f}, acc {100 * d['acc']:.2f}"
                + ("; the frontier is flat above its last vertex" if d.get("flat") else "") + ")")
    return f"{1 - d['w']:.2f} {d['lo']} + {d['w']:.2f} {d['hi']} (x{d['multiple']:.2f}, acc {100 * d['acc']:.2f})"


def line(r, benches):
    if not r.get("judged"):
        return f"{r['name']:34s} {r['kind']:8s} NOT JUDGED: " + (r["reason"] if r.get("layer") == "NOT_JUDGED" else str(r.get("error")))
    e, raw = r["eq"], r["raw"]
    pb = " ".join(f"{short(b)} {100 * e['per_bench'][b]['d']:+5.1f}+-{100 * e['per_bench'][b]['se']:3.1f}{'' if b in r['scored'] else '*'}" for b in benches)
    s = (f"{r['name']:34s} {r['kind']:8s} acc {100 * r['acc']:5.2f} | {'ACCEPT' if r.get('accept') else 'reject'}" + (f" [{r['layer']}]" if r.get("layer") else "")
         + f": {r.get('reason')}\n"
         f"    generated tokens x{r['multiple']:4.2f} | reference {ref_text(r['ref'])} | raw vs 1 tip read {100 * raw['d']:+5.2f} | at equal compute {100 * e['d']:+5.2f} +- {100 * e['se']:4.2f}"
         f" p_macro {e['p_macro']:.4f} p_gen {e['p_gen']:.4f} (worst without {short(e['carrier']) if e['carrier'] else '-'}; t forms {e['p_macro_t']:.4f} / {e['p_gen_t']:.4f}) | benchmark level: t p {e['p_bench_t']:.3f},"
         f" sign-flip p {e['p_bench_signflip']:.3f}, positive {e['npos']}/{e['k']}\n"
         f"    per benchmark (scored {len(r['scored'])}: {' '.join(short(b) for b in r['scored'])}{'; * = source, not scored' if len(r['scored']) < len(benches) else ''}): {pb}")
    t = r.get("total") or {}
    if t.get("d") is not None:
        s += (f"\n    total tokens with prompts x{t['multiple']:4.2f} | reference {ref_text(t['ref'])} | at equal compute {100 * t['d']:+5.2f} +- {100 * t['se']:4.2f}"
              f" p_macro {t['p_macro']:.4f} p_gen {t['p_gen']:.4f} (printed, not gating)")
    return s


def dump(recs, path):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            o = o.item()
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return o
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean(recs), f, indent=1, allow_nan=False)
