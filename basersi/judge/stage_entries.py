import os
import time

from basersi import config as C
from basersi.judge import exports as EX
from basersi.judge import frontier as P
from basersi.judge import guess as G
from basersi.judge import judge as JV
from basersi.judge import reads as R

B = R.BENCHES


def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}"


def sub(arm, ids):
    return {r: arm[r] for r in ids if r in arm}


def frontier_text(F, title):
    return [title] + ["  " + l for l in F.lines()]


def pol_rows(Jg, pols, plain, sig_vertices=None):
    out = []
    V = {p[2] for p in Jg.F.vertices}
    for pol in pols:
        n = P.name_of(pol)
        if n not in Jg.F.table:
            continue
        t = Jg.F.table[n]
        raw = {r: t[r]["ok"] - Jg.tip_v[r] for r in Jg.J}
        blk = Jg._block(raw, B)
        tok = P.macro_mean(t, Jg.rows, "tokens")
        tot = P.macro_mean(t, Jg.rows, "total")
        ref = plain.ref_at(tok)
        eqp = None
        if ref is not None:
            x = {r: t[r]["ok"] - ref[0][r] for r in Jg.J}
            eqp = Jg._block(x, B)
        out.append(dict(name=n, acc=P.macro_mean(t, Jg.rows, "ok"), multiple=tok / Jg.F.base_tokens, multiple_total=tot / Jg.Ft.base_tokens,
                        esc=P.macro_mean(t, Jg.rows, "esc"), raw=blk, vs_plain_votes=eqp, vertex=n in V,
                        vertex_signal=None if sig_vertices is None else n in sig_vertices))
    return out


def pol_line(r):
    e = r["raw"]
    pb = " ".join(f"{b[:5]} {100 * e['per_bench'][b]['d']:+5.1f}" for b in B)
    s = (f"{r['name']:30s} acc {100 * r['acc']:5.2f} tokens x{r['multiple']:4.2f} (total x{r['multiple_total']:4.2f}) esc {100 * r['esc']:4.1f}% |"
         f" raw vs 1 read {100 * e['d']:+5.2f} +- {100 * e['se']:4.2f} p_gen {e['p_gen']:.4f} | {pb} |")
    if r["vs_plain_votes"]:
        v = r["vs_plain_votes"]
        s += f" vs plain votes at equal tokens {100 * v['d']:+5.2f} +- {100 * v['se']:4.2f} p_gen {v['p_gen']:.4f} |"
    s += " VERTEX" if r["vertex"] else " dominated"
    if r["vertex_signal"] is not None:
        s += " (signal rows: " + ("vertex)" if r["vertex_signal"] else "dominated)")
    return s


BUDGET_POLICIES = ("vote1", "n2:disagree->tip", "n3:disagree->tipx2")


def need_votes(judges, row_ids):
    for J_, arm, what in judges:
        missing = [n for n in BUDGET_POLICIES if n not in J_.F.table]
        if missing:
            have = min(len(arm.get(r) or []) for r in row_ids)
            raise JV.JudgeError(f"judge entries: {what} needs at least 5 reads per JUDGE row (the budget point 'n3:disagree->tipx2' = 3 reads + 2 more unless unanimous); "
                             f"fewest reads on a row: {have}; policies missing: {missing}")


def entry_groups(cfg, removal_groups):
    out = {}
    for g in sorted(removal_groups, key=lambda k: list(removal_groups).index(k)):
        spec = (cfg.get("entry_groups") or {}).get(g)
        if spec is not None:
            out[g] = dict(label=spec.get("label", g), entries=list(spec["entries"]))
        elif g.startswith("E") and g[1:].isdigit():
            out[g] = dict(label=g, entries=[int(g[1:])])
        else:
            raise C.ConfigError(f"entry group {g!r} of the removal export has no entry list in judge.yaml entry_groups")
    order = list((cfg.get("entry_groups") or {}).keys())
    return dict(sorted(out.items(), key=lambda kv: (order.index(kv[0]) if kv[0] in order else len(order), kv[0])))


def run(rows_path, export_dir, out_dir, cfg):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    windows = tuple(cfg["guess_policy_windows_entry_audit"])
    st = JV.settings(cfg)
    rows, full = R.load_rows(rows_path), EX.full_rows(rows_path)
    ex = EX.Exports(export_dir, cfg["exports"], cfg.get("sha256"))
    J, S = R.rows_of(rows, "judge"), R.rows_of(rows, R.SIGNAL_ROLES)
    L, doc = ex.reads_lines(), dict(t=time.time(), export_reads=ex.reads)

    tip, fired, ents = ex.tip(rows)
    EX.classify(tip, full)
    floor, stock = ex.floor_and_stock(rows)
    EX.classify(floor, full), EX.classify(stock, full)
    base_arm = ex.base()
    base_text, base_cand = EX.base_label(base_arm)
    doc["base_arm"] = base_arm
    wo, wo_rows = ex.entry_removals(rows)
    for arm in wo.values():
        EX.classify(arm, full)
    src = ex.entry_sources()
    nmax = max(len(v) for v in tip.values())
    short = sum(nmax - len(tip[r]) for r in tip)
    L += ["acceptance tests of the loop tip on the development rows",
          f"loop tip: cached reads of {len(tip)} rows ({sum(len(v) for v in tip.values())} reads; {short} abort records dropped: those rows have {nmax - 1} reads)",
          f"rows: JUDGE {len(J)} ({len(J) // len(B)} per benchmark), SIGNAL {len(S)} (SHOW {len(R.rows_of(rows, 'show'))} + MONITOR {len(R.rows_of(rows, 'monitor'))})",
          "tokens: generated tokens gate; total = generated + prompt tokens of every read, printed beside it", ""]

    tipS, tipJ = sub(tip, S), sub(tip, J)
    Fp = P.Frontier(tipS, S, rows, policies=[dict(n0=1, trigger="none", extra="tip"), dict(n0=2, trigger="disagree", extra="tip", final="plur"),
                                             dict(n0=2, trigger="disagree", extra="tip", final="extra"), dict(n0=3, trigger="none", extra="tip")])
    V = Fp.vertices
    price = (V[1][1] - V[0][1]) / ((V[1][0] - V[0][0]) / V[0][0])
    det = G.derive([dict(row=r, ok=x["ok"], tokens=x["tokens"], tail=x["tail"]) for r in S for x in tipS[r]], rows, price, windows=tuple(cfg["guess_detector_search_windows"]), min_benchmarks=cfg.get("guess_min_benchmarks"))
    JV.dump(det, os.path.join(out_dir, "guess_detector.json"))
    EX.set_guess(tip, det, windows)
    L.append("0. DECLARED-GUESS DETECTOR (derived on SHOW + MONITOR reads)")
    L.append(f"   token price of the tip's own vote on signal rows: {100 * price:.2f} pp per x1 tokens ({V[0][2]} -> {V[1][2]})")
    for W, rec in det["windows"].items():
        sd, lo, fi = rec["seed"], rec["lobo"], rec["final"]
        L.append(f"   tail window {W:4d}: seed 'guess' flags {100 * sd['share']:4.1f}% of reads (accuracy {pct(sd['acc_flagged'])} flagged / {pct(sd['acc_unflagged'])} not);"
                 f" re-reading flagged reads: gain {100 * sd['G']:+.2f} pp at +x{sd['C']:.3f} tokens, value over the vote's price {100 * sd['V']:+.3f}")
        L.append(f"                    + selected tail patterns ({rec['candidates']} candidates): in-sample value {100 * fi['V']:+.3f} with {rec['patterns'][:4]}...;"
                 f" LEAVE-ONE-BENCHMARK-OUT value {100 * lo['V']:+.3f} (better than the seed in {lo['better_in']}/{len(B)} benchmarks) -> "
                 + ("kept" if lo.get("p_signflip", 1.0) <= G.KEEP_P and lo["V"] > sd["V"] else "NOT kept: selected patterns do not transfer across benchmarks"))
    if det["patterns"]:
        L.append(f"   => detector = the stem 'guess' + {det['patterns']} (window {det['window']}); windows {windows} are carried as a policy parameter")
    else:
        L.append(f"   => detector = the stem 'guess' alone (no added pattern survives leave-one-benchmark-out); windows {windows} are carried as a policy parameter")
    L.append("")
    doc["guess_detector"] = dict(window=det["window"], patterns=det["patterns"], price=price,
                                 windows={str(W): dict(seed=r_["seed"], lobo=r_["lobo"], final=r_["final"], patterns=r_["patterns"]) for W, r_ in det["windows"].items()})

    tpols = P.tip_policies(max_vote=5, with_guess=True, windows=windows)
    Js = JV.Judge(rows, tipS, benches=B, policies=tpols, roles=R.SIGNAL_ROLES, settings=st)
    sigV = {p[2] for p in Js.F.vertices}
    sigVt = {p[2] for p in Js.Ft.vertices}
    Jg = JV.Judge(rows, tipJ, benches=B, policies=tpols, hull_from=sigV, hull_from_total=sigVt, settings=st)
    need_votes(((Jg, tipJ, "the tip arm (tip_reads export)"),), J)
    plain_names = ("vote1", "vote3", "vote5")
    plainS = P.Frontier(tipS, S, rows, policies=[Js.F.pols[n] for n in plain_names], table=Js.F.table)
    plainJ = P.Frontier(tipJ, J, rows, policies=[Jg.F.pols[n] for n in plain_names], table=Jg.F.table)
    L.append(f"1. THE TIP'S FRONTIER (tip-only read policies, exact expectation over the {nmax} cached reads; * = vertex). The vertices are CHOSEN on the signal rows")
    L.append("   (upper hull there) and only VALUED on the JUDGE rows: the equal-compute reference of a candidate is the mixture of the two adjacent vertices")
    L.append("   with exactly the candidate's tokens.")
    for nm, Jx in ((f"SIGNAL rows ({len(S)})", Js), (f"JUDGE rows ({len(J)})", Jg)):
        L.append(f"   {nm}: generated tokens | total tokens with prompts")
        Vg, Vt = {p[2] for p in Jx.F.vertices}, {p[2] for p in Jx.Ft.vertices}
        tg = {p[2]: p for p in Jx.F.points}
        tt = {p[2]: p for p in Jx.Ft.points}
        for n in sorted(tg, key=lambda n: tg[n][0]):
            L.append(f"     {n:30s} acc {100 * tg[n][1]:6.2f} | x{tg[n][0] / Jx.F.base_tokens:4.2f} {'*' if n in Vg else ' '} | x{tt[n][0] / Jx.Ft.base_tokens:4.2f} {'*' if n in Vt else ' '}")
    L.append("")
    doc["frontier"] = dict(signal=[dict(name=p[2], tokens=p[0], acc=p[1], vertex=p[2] in sigV) for p in sorted(Js.F.points)],
                           judge=[dict(name=p[2], tokens=p[0], acc=p[1], vertex=p[2] in {q[2] for q in Jg.F.vertices}) for p in sorted(Jg.F.points)],
                           base_tokens=dict(signal=Js.F.base_tokens, judge=Jg.F.base_tokens), base_total=dict(signal=Js.Ft.base_tokens, judge=Jg.Ft.base_tokens))

    L.append(f"2. JUDGE VERDICTS ({len(J)} JUDGE rows). accept iff p_gen passes Holm at {st['alpha']:.2f} in its family, no scored benchmark significantly negative, >= {st['min_scored']} scored benchmarks")
    L.append("")
    spol = P.tip_policies(max_vote=4, with_guess=False)
    Jfs = JV.Judge(rows, sub(stock, S), benches=B, policies=spol, roles=R.SIGNAL_ROLES, settings=st)
    Jf = JV.Judge(rows, sub(stock, J), benches=B, policies=spol, hull_from={p[2] for p in Jfs.F.vertices}, hull_from_total={p[2] for p in Jfs.Ft.vertices}, settings=st)
    plain = not int(base_arm.get("text_entries") or 0)
    rf = Jf.decide([Jf.cand_from_arm(base_cand, floor, "setting" if plain else "general")], family=("image floor" if plain else "general part") + ", K = 1")
    L.append("2a. IMAGE FLOOR against the stock model (reference = the stock model's own frontier from its cached reads)")
    L += frontier_text(Jf.F, "    stock frontier, generated tokens:")
    for r in rf:
        L.append(JV.line(r, B))
    L.append("")
    doc["floor"] = rf

    groups = entry_groups(cfg, wo_rows)
    ac = []
    for k, g in groups.items():
        rr = [r for r in wo_rows[k] if r in Jg.tip]
        srcs = sorted({b for i in g["entries"] for b in src.get(i, [])})
        base = Jg.system_of_arm(wo[k], routed=rr)
        ac.append(Jg.cand_from_arm(g["label"], tipJ, "entry", source=srcs, routed=rr, base=base, unrouted_identical=True))
    ra = Jg.audit_entries(ac)
    L.append("2b. ENTRY AUDIT: each entry (group) as a candidate on top of the tip without it (leave-one-entry-out reads on its routed JUDGE rows;")
    L.append("    tip side = the cached reads). Scored on the benchmarks that did NOT produce it (source = the benchmark of the proposer's card).")
    for r in ra:
        L.append(JV.line(r, B))
        p = r.get("pooled")
        L.append(f"    all {len(B)} benchmarks (source included, if it is one of them): "
                 + (f"{100 * p['d']:+5.2f} +- {100 * p['se']:4.2f} p_macro {p['p_macro']:.4f}" if p else f"not judged ({r.get('reason')})") + f" | routed JUDGE rows {r['routed_rows']}")
    L.append("")
    doc["entry_audit"] = ra

    L.append("3. TIP-ONLY READ POLICIES (they ARE the reference class; no extra read is needed). Raw gain over one read, and over plain votes (1 / 3 / 5) at equal tokens")
    pj = pol_rows(Jg, tpols, plainJ, sigV)
    ps = pol_rows(Js, tpols, plainS)
    L.append("   JUDGE rows:")
    L += ["   " + pol_line(r) for r in sorted(pj, key=lambda r: r["multiple"])]
    L.append("   SIGNAL rows:")
    L += ["   " + pol_line(r) for r in sorted(ps, key=lambda r: r["multiple"])]
    L.append("   increments along the hull (JUDGE rows; is spending the next step of tokens a general gain?):")
    Vj = Jg.F.vertices
    inc = []
    for k in range(1, len(Vj)):
        a_, b_ = Jg.F.table[Vj[k - 1][2]], Jg.F.table[Vj[k][2]]
        blk = Jg._block({r: b_[r]["ok"] - a_[r]["ok"] for r in Jg.J}, B)
        inc.append(dict(lo=Vj[k - 1][2], hi=Vj[k][2], multiple=Vj[k][0] / Jg.F.base_tokens, blk=blk))
        L.append(f"     {Vj[k - 1][2]} -> {Vj[k][2]} (x{Vj[k][0] / Jg.F.base_tokens:.2f}): {100 * blk['d']:+5.2f} +- {100 * blk['se']:4.2f} p_macro {blk['p_macro']:.4f} p_gen {blk['p_gen']:.4f}"
                 f" positive {blk['npos']}/{len(B)} | " + " ".join(f"{b[:5]} {100 * blk['per_bench'][b]['d']:+5.1f}" for b in B))
    doc["tip_policies"] = dict(judge=pj, signal=ps, increments=inc)
    L.append("")

    for xs in floor.values():
        for x in xs:
            x["guess"] = {w: G.flag(dict(det, window=w, patterns=[]), x.get("tail")) for w in windows}
    fS, fJ = sub(floor, S), sub(floor, J)
    Jbs = JV.Judge(rows, fS, benches=B, policies=tpols, roles=R.SIGNAL_ROLES, settings=st)
    Jbg = JV.Judge(rows, fJ, benches=B, policies=tpols, hull_from={p[2] for p in Jbs.F.vertices}, hull_from_total={p[2] for p in Jbs.Ft.vertices}, settings=st)
    L.append(f"4. THE GENERAL BASE = {base_text} and the SPECIALISATION layer on top")
    L.append("   frontier of the general base (vertices chosen on its signal rows, valued on JUDGE rows): generated tokens | total tokens")
    Vg, Vt = {p[2] for p in Jbg.F.vertices}, {p[2] for p in Jbg.Ft.vertices}
    tg = {p[2]: p for p in Jbg.F.points}
    tt = {p[2]: p for p in Jbg.Ft.points}
    ts_ = {p[2]: p for p in Jbs.F.points}
    for n in sorted(tg, key=lambda n: tg[n][0]):
        if n in Vg or n in Vt or n in ("vote1", "vote3", "vote5", "n2:disagree->tip", "n3:disagree->tipx2"):
            L.append(f"     {n:30s} JUDGE acc {100 * tg[n][1]:6.2f} (signal {100 * ts_[n][1]:6.2f}) | x{tg[n][0] / Jbg.F.base_tokens:4.2f} {'*' if n in Vg else ' '} | x{tt[n][0] / Jbg.Ft.base_tokens:4.2f} {'*' if n in Vt else ' '}")
    L.append("   budget points on the JUDGE rows: general base | + specialisation layer (the loop tip = the base + its routed entries) | increment of the layer, per benchmark")
    pts = []
    need_votes(((Jbg, fJ, f"the base arm '{base_arm['arm']}' (floor_and_stock export)"),), J)
    for n, lab in (("vote1", "one read"), ("n2:disagree->tip", "2 reads + third on disagreement"), ("n3:disagree->tipx2", "3 reads + 2 unless unanimous")):
        a_, b_ = Jbg.F.table[n], Jg.F.table[n]
        blk = Jbg._block({r: b_[r]["ok"] - a_[r]["ok"] for r in J}, B)
        raw = Jbg._block({r: a_[r]["ok"] - Jbg.tip_v[r] for r in J}, B)
        ma, mb = P.macro_mean(a_, rows, "ok"), P.macro_mean(b_, rows, "ok")
        pts.append(dict(policy=n, base=ma, base_multiple=P.macro_mean(a_, rows, "tokens") / Jbg.F.base_tokens, with_special=mb,
                        special_multiple=P.macro_mean(b_, rows, "tokens") / Jbg.F.base_tokens, increment=blk, base_gain_over_one_read=raw))
        L.append(f"     {lab:40s} base {100 * ma:6.2f} (x{pts[-1]['base_multiple']:4.2f}; over one read {100 * raw['d']:+5.2f} +- {100 * raw['se']:4.2f}, p_gen {raw['p_gen']:.4f}, positive {raw['npos']}/{len(B)})"
                 f" | with the layer {100 * mb:6.2f} | layer {100 * blk['d']:+5.2f} +- {100 * blk['se']:4.2f} (p_macro {blk['p_macro']:.4f}, p_gen {blk['p_gen']:.4f}, carried by {blk['carrier']}):"
                 + " ".join(f" {b[:5]} {100 * blk['per_bench'][b]['d']:+5.1f}" for b in B))
    a_, b_ = Jbs.F.table["vote1"], Js.F.table["vote1"]
    blk = Jbs._block({r: b_[r]["ok"] - a_[r]["ok"] for r in S}, B)
    L.append(f"   the layer's increment on the SIGNAL rows (one read; rows it was not selected on): {100 * blk['d']:+5.2f} +- {100 * blk['se']:4.2f} (p_macro {blk['p_macro']:.4f}):"
             + " ".join(f" {b[:5]} {100 * blk['per_bench'][b]['d']:+5.1f}" for b in B))
    doc["special_layer_signal_rows"] = blk
    doc["general_base"] = dict(frontier=[dict(name=p[2], tokens=p[0], acc=p[1], vertex=p[2] in Vg) for p in sorted(Jbg.F.points)], points=pts,
                               base_tokens=Jbg.F.base_tokens, signal_vertices=sorted(p[2] for p in Jbs.F.vertices))
    L.append("")
    L.append(f"(computed in {time.time() - t0:.0f} s)")
    open(os.path.join(out_dir, "verdicts.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    JV.dump(doc, os.path.join(out_dir, "verdicts.json"))
    return dict(verdicts=os.path.join(out_dir, "verdicts.json"), floor_accepted=bool(rf[0]["accept"]),
                entry_layers={r["name"]: r["layer"] for r in ra}, guess_detector=os.path.join(out_dir, "guess_detector.json"))
