import collections
import hashlib
import json
import math
import os
import statistics
import time

from basersi import config as C
from basersi.judge import exports as EX
from basersi.judge import frontier as P
from basersi.judge import judge as JV
from basersi.judge import reads as R
from basersi.judge import stage_view_edit as S4

B = R.BENCHES
NAME = "Procedure Edit"
BASE_NAME = "View Edit"
TIP_FILE = "tip_procedure_edit.json"
TITLE = "View Edit + route-keyed Procedure Edit cell"
UNITS_KIND = "route-keyed MLP cell compiled from the admitted procedure entry (its gate and step are in the Procedure Edit spec); in the weights, gated inside the model"


class Context(S4.Context):
    def __init__(self, rows_path, export_dir, out_dir, cfg):
        super().__init__(rows_path, export_dir, out_dir, cfg)
        if cfg.get("procedure_edit_source_benchmark") is None or cfg.get("procedure_edit_entry_index") is None:
            raise C.MissingInput("procedure edit acceptance: the source benchmark and the index of the compiled entry are unknown. They come from judge[entries] (judge.procedure_entry); "
                             "when that step chose no entry (none labelled ID_SPECIFIC), set judge.procedure_edit_source_benchmark and judge.procedure_edit_entry_index in the pipeline "
                             "config (or --source-benchmark / --entry-index of `judge procedure-edit`)")
        self.source_bench = cfg["procedure_edit_source_benchmark"]
        self.cell_entry = int(cfg["procedure_edit_entry_index"])


def base_judges(ctx, want_judge):
    a_s, _ = S4.load_view_edit(ctx, "view_edit_signal", ctx.S)
    pols = P.tip_policies(max_vote=4, with_guess=False)
    Ws = JV.Judge(ctx.rows, a_s, benches=B, policies=pols, roles=R.SIGNAL_ROLES, settings=ctx.settings)
    Wg = None
    if want_judge:
        a_j, _ = S4.load_view_edit(ctx, "view_edit_judge", ctx.J)
        Wg = JV.Judge(ctx.rows, a_j, benches=B, policies=pols, hull_from={p[2] for p in Ws.F.vertices}, hull_from_total={p[2] for p in Ws.Ft.vertices},
                      settings=ctx.settings)
    return Ws, Wg


def point(Jx, lab, pol, arm):
    c = Jx.cand_from_views(f"{NAME} {lab}", pol, {NAME: arm}, kind="weights")
    n = len(Jx.J)
    gen, tot = sum(c["tokens"].values()) / n, sum(c["total"].values()) / n
    r = Jx.score(c)
    out = dict(budget=lab, acc=sum(c["value"].values()) / n, gen=gen, prompt=tot - gen, total=tot, multiple=gen / Jx.base_tokens, multiple_total=tot / Jx.base_total,
               judged=bool(r.get("judged")))
    if not r.get("judged"):
        out.update(error=r.get("error"), **{k: r[k] for k in JV.ABOVE_KEYS if k in r})
        return out, c, r
    e, t = r["eq"], r.get("total") or {}
    out.update(raw=r["raw"]["d"], raw_se=r["raw"]["se"], d=e["d"], se=e["se"], p_macro=e["p_macro"], p_gen=e["p_gen"], carrier=e["carrier"], npos=e["npos"], neg=[b for b, _ in e["neg"]],
               per_bench={b: e["per_bench"][b]["d"] for b in B}, per_bench_se={b: e["per_bench"][b]["se"] for b in B},
               ref=r["ref"], d_total=t.get("d"), se_total=t.get("se"), p_gen_total=t.get("p_gen"), ref_total=t.get("ref"))
    return out, c, r


def subset(Jx, x, rs):
    rs = [r for r in rs if r in x]
    if not rs:
        return None
    cl = {}
    for r in rs:
        cl.setdefault(Jx.rows[r]["pic"], []).append(x[r])
    n, G = len(rs), len(cl)
    d = sum(x[r] for r in rs) / n
    se = math.sqrt(sum((sum(v) - d * len(v)) ** 2 for v in cl.values()) * G / max(G - 1, 1)) / n
    return dict(rows=n, d=d, se=se)


def route_split(Jx, c, routed):
    x, info = Jx.diffs(c)
    out = {}
    for lab, rs in (("routed", [r for r in Jx.J if r in routed]), ("not routed", [r for r in Jx.J if r not in routed])):
        out[lab] = dict(eq=subset(Jx, x, rs) if x is not None else None, raw=subset(Jx, info["raw"], rs) if x is not None else None,
                        by_bench=dict(collections.Counter(Jx.rows[r]["bench"] for r in rs)))
    return out


def off_source(Jx, c, source_bench):
    x, _ = Jx.diffs(c)
    if x is None:
        return None
    S5 = [b for b in B if b != source_bench]
    blk = Jx._block(x, S5)
    return dict(scored=S5, d=blk["d"], se=blk["se"], p_macro=blk["p_macro"], p_gen=blk["p_gen"], npos=blk["npos"], carrier=blk["carrier"])


def tails(arm, ref_arm, rows_):
    c = [x["tokens"] for r in rows_ for x in arm[r]]
    w = [x["tokens"] for r in rows_ for x in ref_arm[r]]
    q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))]
    return dict(mean=[sum(c) / len(c), sum(w) / len(w)], median=[statistics.median(c), statistics.median(w)], p95=[q(c, .95), q(w, .95)], max=[max(c), max(w)],
                not_stop=[sum(x.get("finish") != "stop" for r in rows_ for x in arm[r]), sum(x.get("finish") != "stop" for r in rows_ for x in ref_arm[r])],
                reads=[len(c), len(w)])


def gate_info(Jx, arm, routed, c=None):
    out = {}
    raw = Jx.diffs(c)[1].get("raw") if c is not None else None
    mt = lambda vs: sum(v["tokens"] for v in vs) / len(vs)
    for lab, rs in (("routed", [r for r in Jx.J if r in routed]), ("not routed", [r for r in Jx.J if r not in routed])):
        if not rs:
            continue
        cc = [x["tokens"] for r in rs for x in arm[r]]
        w = [x["tokens"] for r in rs for x in Jx.tip[r]]
        g = dict(rows=len(rs), mean=[sum(cc) / len(cc), sum(w) / len(w)], median=[statistics.median(cc), statistics.median(w)])
        g["tok"] = subset(Jx, {r: mt(arm[r]) - mt(Jx.tip[r]) for r in rs}, rs)
        g["log"] = subset(Jx, {r: math.log(mt(arm[r])) - math.log(mt(Jx.tip[r])) for r in rs}, rs)
        aa = {}
        for r in rs:
            u = sorted(Jx.tip[r], key=lambda v: v["s"])
            if len(u) >= 4:
                aa[r] = (mt(u[:2]), mt(u[2:4]))
        if len(aa) == len(rs):
            g["aa_tok"] = subset(Jx, {r: v[0] - v[1] for r, v in aa.items()}, rs)
            g["aa_log"] = subset(Jx, {r: math.log(v[0]) - math.log(v[1]) for r, v in aa.items()}, rs)
        if raw is not None:
            g["by_bench"] = {b_: subset(Jx, raw, [r for r in rs if Jx.rows[r]["bench"] == b_]) for b_ in B}
        out[lab] = g
    return out


def agreement(arm, rows_):
    ag = [(P.cls_of(arm[r][0]) == P.cls_of(arm[r][1]), (arm[r][0]["ok"] + arm[r][1]["ok"]) / 2) for r in rows_ if len(arm[r]) >= 2]
    a, d_ = [v for k, v in ag if k], [v for k, v in ag if not k]
    return dict(rows=len(ag), agree=len(a) / max(len(ag), 1), acc_agree=sum(a) / max(len(a), 1), acc_disagree=sum(d_) / max(len(d_), 1))


def same_index(Jx, arm, routed):
    x, ident = {}, {"routed": [0, 0], "not routed": [0, 0]}
    for r in Jx.J:
        w = {v["s"]: v for v in Jx.tip[r]}
        ds = []
        for v in arm[r]:
            if v["s"] in w:
                u = w[v["s"]]
                ds.append(float(v["ok"]) - float(u["ok"]))
                k = "routed" if r in routed else "not routed"
                ident[k][1] += 1
                ident[k][0] += int(P.cls_of(v) == P.cls_of(u) and v["tokens"] == u["tokens"])
        if ds:
            x[r] = sum(ds) / len(ds)
    if len(x) < len(Jx.J):
        return None
    blk = Jx._block(x, B)
    return dict(d=blk["d"], se=blk["se"], p_macro=blk["p_macro"], p_gen=blk["p_gen"], per_bench={b: blk["per_bench"][b]["d"] for b in B},
                identical={k: (v[0] / v[1] if v[1] else None) for k, v in ident.items()}, pairs={k: v[1] for k, v in ident.items()},
                routed=subset(Jx, x, [r for r in Jx.J if r in routed]), not_routed=subset(Jx, x, [r for r in Jx.J if r not in routed]))


def compute(ctx, Jx, arm, routed):
    pts = [point(Jx, "one read", P.views_policy([NAME]), arm)]
    if min(len(arm[r]) for r in Jx.J) >= 2:
        pts.append(point(Jx, f"2 reads + a {BASE_NAME} read on disagreement", P.views_policy([NAME, NAME], ["tip"]), arm))
    return pts, route_split(Jx, pts[0][1], routed), tails(arm, Jx.tip, Jx.J), dict(agreement(arm, Jx.J), same_index=same_index(Jx, arm, routed),
                                                                                    off_source=off_source(Jx, pts[0][1], ctx.source_bench),
                                                                                    gate=gate_info(Jx, arm, routed, pts[0][1]))


def lines_C(ctx, L, Jx, pts, split, tl, ag, title):
    L += ["", title, S4.base_line(Jx, f"{BASE_NAME} ({S4.reads_per_row(Jx.tip)} reads per row)")]
    for x, _, _ in pts:
        L.append(S4.text(x))
    L.append(f"  split of the ONE-READ effect by the route of the cell's source entry (the loop tip's cached routes):")
    for lab, v in split.items():
        if v["eq"]:
            L.append(f"    {lab:11s} {v['eq']['rows']:3d} rows: raw vs 1 {BASE_NAME} read {100 * v['raw']['d']:+6.2f} +- {100 * v['raw']['se']:4.2f} | at equal generated tokens {100 * v['eq']['d']:+6.2f} +- {100 * v['eq']['se']:4.2f} | rows by benchmark {v['by_bench']}")
    gt = ag.get("gate")
    if gt:
        L.append(f"  THE GATE (information): generated tokens {NAME} / {BASE_NAME} by route: " + "; ".join(
            f"{lab} ({v['rows']} rows) mean {v['mean'][0]:.0f} / {v['mean'][1]:.0f} (x{v['mean'][0] / v['mean'][1]:.3f}), median {v['median'][0]:.0f} / {v['median'][1]:.0f} (x{v['median'][0] / v['median'][1]:.3f})"
            for lab, v in gt.items()))
    for lab, v in (gt or {}).items():
        if v.get("tok"):
            t_, l_ = v["tok"], v["log"]
            x_ = (f"    {lab}: paired by row, {NAME} minus {BASE_NAME} generated tokens {t_['d']:+.0f} +- {t_['se']:.0f} ({100 * t_['d'] / v['mean'][1]:+.1f} +- {100 * t_['se'] / v['mean'][1]:.1f} % of {BASE_NAME}'s mean);"
                  f" ratio of the row means, geometric x{math.exp(l_['d']):.3f} (log {l_['d']:+.3f} +- {l_['se']:.3f})")
            if v.get("aa_tok"):
                x_ += (f" | {BASE_NAME} against itself (reads s0, s1 minus s2, s3): {v['aa_tok']['d']:+.0f} +- {v['aa_tok']['se']:.0f} tokens, geometric x{math.exp(v['aa_log']['d']):.3f}"
                       f" (log {v['aa_log']['d']:+.3f} +- {v['aa_log']['se']:.3f})")
            L.append(x_)
        if v.get("by_bench"):
            L.append(f"    {lab}: raw one read minus one {BASE_NAME} read by benchmark: " + " ".join(
                f"{b_[:5]} {100 * e['d']:+.1f}+-{100 * e['se']:.1f} ({e['rows']})" for b_, e in v["by_bench"].items() if e))
    L.append(f"  generated tokens {NAME} / {BASE_NAME}: mean {tl['mean'][0]:.0f} / {tl['mean'][1]:.0f} (x{tl['mean'][0] / tl['mean'][1]:.2f}), median {tl['median'][0]:.0f} / {tl['median'][1]:.0f}"
             f" (x{tl['median'][0] / tl['median'][1]:.2f}), 95th percentile {tl['p95'][0]:.0f} / {tl['p95'][1]:.0f}, max {tl['max'][0]:.0f} / {tl['max'][1]:.0f},"
             f" reads not ending in stop {tl['not_stop'][0]} of {tl['reads'][0]} / {tl['not_stop'][1]} of {tl['reads'][1]}")
    L.append(f"  the two {NAME} reads agree on {100 * ag['agree']:.1f}% of rows (accuracy {100 * ag['acc_agree']:.1f} when they agree, {100 * ag['acc_disagree']:.1f} when they do not)")
    of = ag.get("off_source")
    if of:
        L.append(f"  OFF-SOURCE (the general-layer rule): one read scored WITHOUT {ctx.source_bench} (the source benchmark of the entry the cell was compiled from):"
                 f" {100 * of['d']:+.2f} +- {100 * of['se']:.2f} over the other {len(B) - 1} benchmarks, p_macro {of['p_macro']:.4f}, p_gen {of['p_gen']:.4f}, positive {of['npos']}/{len(B) - 1}")
    si = ag.get("same_index")
    if si:
        L.append(f"  printed, not deciding: {NAME} read s minus the {BASE_NAME} read of the same row and sample index (common seeds): {100 * si['d']:+.2f} +- {100 * si['se']:.2f} (p_macro {si['p_macro']:.4f}, p_gen {si['p_gen']:.4f});"
                 f" routed {pm(si['routed'])}, not routed {pm(si['not_routed'])};"
                 f" the two reads are the SAME read (same answer, same token count) on {100 * (si['identical']['routed'] or 0):.1f}% of routed and {100 * (si['identical']['not routed'] or 0):.1f}% of unrouted (row, sample) pairs")


def pm(e):
    return f"{100 * e['d']:+.2f} +- {100 * e['se']:.2f}" if e else "n/a (no rows)"


def info_D(Jx, arm, tip):
    m = len(Jx.J)
    v = {r: sum(x["ok"] for x in arm[r]) / len(arm[r]) for r in Jx.J}
    tv = {r: sum(x["ok"] for x in tip[r]) / len(tip[r]) for r in Jx.J}
    xd = {r: v[r] - tv[r] for r in Jx.J}
    blk = Jx._block(xd, B)
    mean = lambda a, k: sum(sum(k(x) for x in a[r]) / len(a[r]) for r in Jx.J) / m
    g = [mean(arm, lambda x: x["tokens"]), mean(tip, lambda x: x["tokens"])]
    p_ = [mean(arm, lambda x: float(x.get("prompt_tokens") or 0)), mean(tip, lambda x: float(x.get("prompt_tokens") or 0))]
    return dict(acc=sum(v.values()) / m, acc_tip=sum(tv.values()) / m, d=blk["d"], se=blk["se"], p_two_sided=Jx.p_two_sided(xd, B), npos=blk["npos"],
                per_bench={b: blk["per_bench"][b]["d"] for b in B}, per_bench_se={b: blk["per_bench"][b]["se"] for b in B}, gen=g, prompt=p_, total=[g[0] + p_[0], g[1] + p_[1]])


def text_D(x):
    return (f"  one read: {NAME} {100 * x['acc']:.2f} vs loop tip {100 * x['acc_tip']:.2f}: {100 * x['d']:+.2f} +- {100 * x['se']:.2f} (two-sided p {x['p_two_sided']:.3f}; positive {x['npos']}/{len(B)}) | "
            + " ".join(f"{b[:5]} {100 * x['per_bench'][b]:+5.1f}+-{100 * x['per_bench_se'][b]:3.1f}" for b in B)
            + f"\n  tokens {NAME} / loop tip: generated {x['gen'][0]:.0f} / {x['gen'][1]:.0f} (x{x['gen'][0] / x['gen'][1]:.2f}), prompt {x['prompt'][0]:.0f} / {x['prompt'][1]:.0f} (x{x['prompt'][0] / x['prompt'][1]:.2f}),"
            f" total {x['total'][0]:.0f} / {x['total'][1]:.0f} (x{x['total'][0] / x['total'][1]:.2f})")


def load_all(ctx, key, role_rows):
    arm, meta = S4.load_candidate(ctx.ex.path(key), ctx.rows, ctx.full, role_rows, ctx.cfg["exports"][key]["arm_hint"])
    tip, fired, _ = ctx.ex.tip(ctx.rows)
    EX.classify(tip, ctx.full)
    routed = {r for r, f in fired.items() if ctx.cell_entry in f}
    return arm, meta, tip, routed


def run_signal(ctx):
    os.makedirs(ctx.out_dir, exist_ok=True)
    t0 = time.time()
    arm, meta, tip, routed = load_all(ctx, "procedure_edit_signal", ctx.S)
    Ws, _ = base_judges(ctx, False)
    pts, split, tl, ag = compute(ctx, Ws, arm, routed)
    Dm = info_D(Ws, arm, tip)
    L = [f"{NAME} = {TITLE} on the SIGNAL rows ({len(ctx.S)} rows; they include the cell's fit rows): the signal computation; never a decision",
         f"candidate reads: {json.dumps(meta)}"]
    lines_C(ctx, L, Ws, pts, split, tl, ag, f"C. against the {BASE_NAME} system's own frontier (frontier against frontier at equal generated tokens)")
    se_ = Ws.source_effect(pts[0][1], [ctx.source_bench])
    L.append(f"  for the record (signal rows hold the cell's fit rows): the admission form on the {ctx.source_bench} signal rows: "
             + (f"{100 * se_['d']:+.2f} +- {100 * se_['se']:.2f} (p {se_['p']:.4f}; in-sample)" if se_.get("judged") else f"not judged: {se_.get('error')}"))
    L += ["", "D. INFORMATION: against the loop tip (request floor + text entries; cached reads; decides nothing)", text_D(Dm)]
    L.append(f"\n({time.time() - t0:.0f} s)")
    open(os.path.join(ctx.out_dir, "procedure_edit_signal.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    path = os.path.join(ctx.out_dir, "procedure_edit_signal.json")
    JV.dump(dict(meta=meta, C=[x for x, _, _ in pts], split=split, tokens=tl, agreement=ag, D=Dm, judged=bool(pts[0][0]["judged"])), path)
    return dict(signal=path)


def run_judge(ctx):
    t0 = time.time()
    src = ctx.source_bench
    arm, meta, tip, routed = load_all(ctx, "procedure_edit_judge", ctx.J)
    sig = json.load(open(os.path.join(ctx.out_dir, "procedure_edit_signal.json"), encoding="utf-8"))
    Ws, Wg = base_judges(ctx, True)
    pts, split, tl, ag = compute(ctx, Wg, arm, routed)
    cand = pts[0][1]
    rec = Wg.decide([dict(cand, source=[src])], family=f"{NAME} general layer, off-source, K = 1")[0]
    if not rec.get("judged"):
        return not_judged(ctx, t0, meta, sig, rec, Wg, arm, tip, pts, split, tl, ag)
    acc = bool(rec["accept"])
    allb = pts[0][0]
    adm = Wg.source_effect(cand, [src]) if not acc else None
    src_ok = bool(adm and adm.get("judged") and adm["d"] > 0 and adm["p"] <= ctx.settings['alpha'])
    admitted = bool(src_ok and not allb.get("neg"))
    why_not = None if (admitted or acc) else ("significantly negative on " + ", ".join(allb["neg"]) if (src_ok and allb.get("neg")) else "no significant gain on the source benchmark")
    Dm = info_D(Wg, arm, tip)
    e5 = rec["eq"]
    L = [f"VERDICT for {NAME} = {TITLE} ({UNITS_KIND}; source benchmark {src}; stock requests, no text)",
         f"{len(ctx.J)} JUDGE rows; candidate reads: {json.dumps(meta)}", *ctx.ex.reads_lines(), "",
         f"GENERAL-LAYER VERDICT (off-source: {len(B) - 1} benchmarks without {src}; one read against {BASE_NAME}'s frontier at equal generated tokens; K = 1): "
         f"{'ACCEPTED' if acc else 'NOT ACCEPTED'}: {100 * e5['d']:+.2f} +- {100 * e5['se']:.2f}, p_macro {e5['p_macro']:.4f}, p_gen {e5['p_gen']:.4f}, positive {e5['npos']}/{len(B) - 1}; {rec['reason']}",
         f"  information, all {len(B)} benchmarks scored: {100 * allb['d']:+.2f} +- {100 * allb['se']:.2f}, p_macro {allb['p_macro']:.4f}, p_gen {allb['p_gen']:.4f} (worst without {allb['carrier']}), positive {allb['npos']}/{len(B)}"]
    if adm is not None:
        L.append(f"ADMISSION TEST (the source benchmark's own JUDGE rows, family K = 1, Holm {ctx.settings['alpha']:.2f}, sign-flip of picture clusters, equal generated tokens; AND no benchmark significantly negative): "
                 + (f"{'ADMITTED as a specialisation component held in weights (ID_SPECIFIC)' if admitted else 'NOT ADMITTED (' + str(why_not) + ')'}: {src} {100 * adm['d']:+.2f} +- {100 * adm['se']:.2f} on its {adm['rows']} JUDGE rows"
                    f" (raw {100 * adm['raw']:+.2f}), p {adm['p']:.4f}; significantly negative benchmarks: {allb.get('neg') or 'none'}"
                    + ("".join(f" ({b} {100 * allb['per_bench'][b]:+.1f} +- {100 * allb['per_bench_se'][b]:.1f})" for b in allb.get("neg") or [])) if adm.get("judged") else f"not judged: {adm.get('error')}"))
        nr = split.get("not routed") or {}
        gt = ag.get("gate") or {}
        if nr.get("raw") and gt.get("not routed"):
            g_ = gt["not routed"]
            L.append(f"  THE GATE (information): the {nr['raw']['rows']} JUDGE rows where the source entry does not route: {NAME} minus {BASE_NAME} {100 * nr['raw']['d']:+.2f} +- {100 * nr['raw']['se']:.2f} raw"
                     f" ({100 * nr['eq']['d']:+.2f} at equal generated tokens); generated tokens median {g_['median'][0]:.0f} vs {g_['median'][1]:.0f} (x{g_['median'][0] / g_['median'][1]:.3f}),"
                     f" mean x{g_['mean'][0] / g_['mean'][1]:.3f}")
        L.append(f"  rows that did not select it: the cell was fitted on SIGNAL rows only, so these {src} JUDGE rows are themselves untouched by the fit; its signal-row number is in-sample"
                 f" ({src} signal rows " + (f"{100 * sig['C'][0]['per_bench'][src]:+.2f}" if sig["C"][0].get("judged") else f"not judged: {sig['C'][0].get('error')}") + ").")
    lines_C(ctx, L, Wg, pts, split, tl, ag, f"C. against the {BASE_NAME} system's own frontier (frontier against frontier at equal generated tokens; all {len(B)} benchmarks scored)")
    sd, jd = sig["C"][0], pts[0][0]
    kept = jd["d"] / sd["d"] if (sd.get("judged") and jd.get("judged") and sd["d"] > 0) else None
    L += ["", "SHRINKAGE for C (one read, equal generated tokens): signal rows "
          + (f"{100 * sd['d']:+.2f} +- {100 * sd['se']:.2f} (raw {100 * sd['raw']:+.2f}; in-sample on the cell's fit rows)" if sd.get("judged") else f"not judged ({sd.get('error')})")
          + f" -> JUDGE rows {100 * jd['d']:+.2f} +- {100 * jd['se']:.2f} (raw {100 * jd['raw']:+.2f})" + (f"; kept {100 * kept:.0f}%" if kept is not None else "")
          + "; routed rows raw: signal " + routed_raw(sig.get("split")) + " -> JUDGE " + routed_raw(split)]
    L += ["", f"D. INFORMATION (decides nothing): the weights-only system {NAME} against the text-and-settings loop tip (request floor + text entries), one read", text_D(Dm)]
    out = dict(meta=meta, export_reads=ctx.ex.reads, candidate=NAME, judged=True, accepted_general=acc, admitted_specialisation=admitted, not_admitted_because=why_not, reason=rec["reason"], C=[x for x, _, _ in pts], split=split, tokens=tl, agreement=ag, D=Dm, signal=sig,
               shrinkage=dict(signal=sd.get("d"), judge=jd.get("d"), kept=kept),
               decision=dict(point="one read", rule=f"off-source ({len(B) - 1} benchmarks), K = 1", scored=rec["scored"], p_gen=e5["p_gen"], p_macro=e5["p_macro"], d=e5["d"], se=e5["se"], K=1,
                             alpha=ctx.settings['alpha'], neg=[b for b, _ in e5["neg"]], all_benchmarks=dict(d=allb["d"], se=allb["se"], p_gen=allb["p_gen"], p_macro=allb["p_macro"])),
               admission=adm)
    parent, no_parent = view_edit_tip(ctx)
    if parent is not None:
        out["tip_file"] = write_tip(ctx, parent, meta, sig, rec, pts, split, general=acc, adm=adm, admitted=admitted, why_not=why_not)
        L += ["", f"TIP FILE ({'general' if acc else 'admitted: specialisation layer in weights' if admitted else 'NOT ADMITTED: kept as a record, the system stays the View Edit system'}): {out['tip_file']}"]
    else:
        out["tip_file"] = None
        L += ["", f"TIP FILE: none ({no_parent}; the {NAME} tip is built on the View Edit tip)"]
        set_aside_tip(ctx, L)
    L.append(f"\n({time.time() - t0:.0f} s)")
    open(os.path.join(ctx.out_dir, "procedure_edit_judge.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    path = os.path.join(ctx.out_dir, "procedure_edit_judge.json")
    JV.dump(out, path)
    return dict(judge=path, judged=True, accepted_general=acc, admitted_specialisation=admitted, tip_file=out["tip_file"])


def routed_raw(split):
    raw = ((split or {}).get("routed") or {}).get("raw")
    return f"{100 * raw['d']:+.2f}" if raw else "n/a"


def set_aside_tip(ctx, L):
    old = os.path.join(ctx.out_dir, TIP_FILE)
    if os.path.exists(old):
        moved = R.superseded_name(old)
        os.replace(old, moved)
        L.append(f"the {TIP_FILE} of an earlier run was moved to {os.path.basename(moved)}")


def not_judged(ctx, t0, meta, sig, rec, Wg, arm, tip, pts, split, tl, ag):
    why = rec.get("error") or rec.get("reason") or "no comparison with the reference frontier"
    ct, ft = rec.get("candidate_tokens"), rec.get("frontier_max_tokens")
    figs = (f"; the candidate generates {ct:.0f} tokens per row, the frontier covers up to {ft:.0f}" if ct is not None and ft is not None and not rec.get("above_frontier") else "")
    shown = why[len("NOT JUDGED: "):] if why.startswith("NOT JUDGED: ") else why
    L = [f"VERDICT for {NAME} = {TITLE} ({UNITS_KIND}; source benchmark {ctx.source_bench}; stock requests, no text)",
         f"{len(ctx.J)} JUDGE rows; candidate reads: {json.dumps(meta)}", *ctx.ex.reads_lines(), "",
         f"NOT JUDGED (neither accepted nor admitted; no tip file): {shown}{figs}. The pipeline stops here; no setting turns this into a pass."]
    lines_C(ctx, L, Wg, pts, split, tl, ag, f"C. against the {BASE_NAME} system's own frontier (information only: the candidate is outside it)")
    L += ["", f"D. INFORMATION (decides nothing): the weights-only system {NAME} against the text-and-settings loop tip (request floor + text entries), one read", text_D(info_D(Wg, arm, tip))]
    L += ["", "TIP FILE: none (not judged)"]
    set_aside_tip(ctx, L)
    L.append(f"\n({time.time() - t0:.0f} s)")
    open(os.path.join(ctx.out_dir, "procedure_edit_judge.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    path = os.path.join(ctx.out_dir, "procedure_edit_judge.json")
    res = dict(judged=False, reason="NOT JUDGED: " + shown, candidate_tokens=ct, frontier_max_tokens=ft, frontier_max_multiple=rec.get("frontier_max_multiple"),
               above_frontier=bool(rec.get("above_frontier")), accepted_general=False, admitted_specialisation=False, tip_file=None)
    JV.dump(dict(res, meta=meta, export_reads=ctx.ex.reads, candidate=NAME, C=[x for x, _, _ in pts], split=split, tokens=tl, agreement=ag, signal=sig), path)
    return dict(res, judge=path)


def view_edit_tip(ctx):
    p = os.path.join(ctx.out_dir, S4.TIP_FILE)
    if not os.path.exists(p):
        return None, f"{S4.TIP_FILE} is missing: the View Edit was not accepted"
    parent = json.load(open(p, encoding="utf-8"))
    current = hashlib.sha256(open(ctx.ex.path("view_edit_judge"), "rb").read()).hexdigest()
    judged_on = ((parent.get("reads") or {}).get("judge") or {}).get("sha256")
    if judged_on != current:
        return None, f"{S4.TIP_FILE} was written for other View Edit reads ({str(judged_on)[:16]}...) than the ones judged now ({current[:16]}...)"
    return parent, None


def write_tip(ctx, parent, meta, sig, rec, pts, split, general, adm, admitted, why_not):
    src = ctx.source_bench
    w = [json.loads(x) for x in meta["weights"]]
    art = w[0] if len(w) == 1 else w
    one = pts[0][0]
    units = dict(name="procedure_edit", always_on=False, kind=UNITS_KIND)
    v = dict(cid=units["name"], benches=list(B), scored=rec["scored"], d=rec["eq"]["d"], se=rec["eq"]["se"], p_gen=rec["eq"]["p_gen"],
             all_benchmarks=dict(d=one.get("d"), se=one.get("se"), p_gen=one.get("p_gen")), one_read_raw=one.get("raw"),
             routed_raw=(split["routed"]["raw"] or {}).get("d"), not_routed_raw=(split["not routed"]["raw"] or {}).get("d"),
             negative_benchmarks=list(one.get("neg") or []), per_bench=one.get("per_bench"), per_bench_se=one.get("per_bench_se"))
    if adm and adm.get("judged"):
        v.update(source_d=adm["d"], source_se=adm["se"], source_p=adm["p"], source_raw=adm["raw"], source_rows=adm["rows"], K=1,
                 untouched_rows="the cell was fitted on signal rows only: the source benchmark's JUDGE rows did not select it",
                 signal_source_d_in_sample=(sig["C"][0].get("per_bench") or {}).get(src))
    cell = dict(condition="always", cond="always", procedure=None, settings={}, ops=[], weights=art, cid=units["name"], units=units, source=[src],
                author="loop (compiled cell; pre-declared)")
    n_reads = int(round(meta["lines"] / meta["rows"])) if meta.get("rows") else None
    spec_sys = dict(artefact=art, one_read_judge=one["acc"], one_read_signal=sig["C"][0]["acc"], reads_per_row=n_reads, tokens=dict(generated=one["gen"], prompt=one["prompt"], total=one["total"]))
    entries, special, rejected, policy, system = list(parent["entries"]), [], [], dict(parent["policy"]), None
    if general:
        status = "general"
        entries.append(dict(cell, audit="GENERAL", verdict=dict(v, layer="GENERAL", basis=f"off-source ({len(B) - 1} benchmarks), one read against the {BASE_NAME} frontier at equal generated tokens, K = 1")))
        note = f"General layer = the View Edit weights + the {NAME} cell (it passed the off-source rule), stock requests, no text; the specialisation layer is empty."
        policy = dict(default=parent["policy"]["default"], extras=[], guess=None,
                      frontier=dict(vertices=["vote1"], points=[dict(name="vote1", vertex=True, multiple=1.0, multiple_total=1.0, acc_judge=one["acc"], acc_signal=sig["C"][0]["acc"])],
                                    default="vote1", report=["vote1"], base_tokens=one["gen"], base_total=one["total"], reads_per_row=n_reads,
                                    note=f"{n_reads} reads per row: only the one-read point is valued; the default budget and the other vertices need more reads per row"))
    elif admitted:
        status = "admitted: specialisation layer in weights (ID_SPECIFIC)"
        special.append(dict(cell, held_in="weights", verdict=dict(v, layer="ID_SPECIFIC", basis=f"admission test: the source benchmark's JUDGE rows (K = 1, Holm {ctx.settings['alpha']:.2f}) and no benchmark significantly negative")))
        system = spec_sys
        note = (f"General layer = the View Edit weights only. SPECIALISATION layer IN WEIGHTS = the {NAME} cell (source benchmark {src}; it failed the off-source rule and passed "
                "the admission test: positive on its source benchmark's JUDGE rows and no benchmark significantly negative). The SERVED ARTEFACT CARRIES BOTH; the GENERAL system's reported "
                "numbers are those of the View Edit system (the artefact without the cell). The artefact with the cell is the specialisation system, reported separately, never as a general gain.")
    else:
        status = f"not admitted ({why_not})"
        rejected.append(dict(cell, verdict=dict(v, layer="NOT_ADMITTED", why=why_not, basis=f"general layer: off-source rule failed; admission: the source benchmark's JUDGE rows (K = 1, Holm {ctx.settings['alpha']:.2f}) AND no "
                                                "benchmark significantly negative (a component in the weights can touch every row)")))
        note = (f"NOT ADMITTED. The {NAME} cell is neither in the general layer (off-source rule failed) nor in the specialisation layer ({why_not}). The system stays the View Edit system; "
                "this file is the record of the verdict with every number.")
    tip = dict(id="procedure_edit", status=status, parent="view_edit", kind="weights system", note=note,
               weights=dict(name=f"{NAME} = {TITLE}", artefact=art, general_artefact=parent["weights"]["artefact"], requests="stock requests: no min_image_tokens floor"),
               entries=entries, special=special, rejected=rejected, policy=policy, specialisation_system=system, candidate_artefact_reads=spec_sys,
               reads=dict(signal=sig["meta"], judge=meta))
    p = os.path.join(ctx.out_dir, TIP_FILE)
    JV.dump(tip, p)
    return p
