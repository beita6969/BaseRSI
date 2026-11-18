import collections
import gzip
import hashlib
import json
import os
import time

from basersi.judge import exports as EX
from basersi.judge import frontier as P
from basersi.judge import guess as G
from basersi.judge import judge as JV
from basersi.judge import reads as R

B = R.BENCHES
NAME = "View Edit"
TIP_FILE = "tip_view_edit.json"


class Context:
    def __init__(self, rows_path, export_dir, out_dir, cfg):
        self.rows_path, self.out_dir, self.cfg = rows_path, out_dir, cfg
        self.ex = EX.Exports(export_dir, cfg["exports"], cfg.get("sha256"))
        self.windows = tuple(cfg["guess_policy_windows_weight_edits"])
        self.settings = JV.settings(cfg)
        self.prescreen_p = float(cfg["prescreen_p"])
        self.default_policy = cfg["default_read_policy"]
        self.rows = R.load_rows(rows_path)
        self.full = EX.full_rows(rows_path)
        self.S = R.rows_of(self.rows, R.SIGNAL_ROLES)
        self.J = R.rows_of(self.rows, "judge")


def budgets(view, n):
    out = [("one read", P.views_policy([view])), ("default budget", P.views_policy([view, view], [view]))]
    if n >= 5:
        out.append(("3 reads + 2 unless unanimous", P.views_policy([view, view, view], [view, view])))
    elif n >= 4:
        out.append((f"vote of {n}", P.views_policy([view] * n)))
    return out


def load_candidate(path, rows, full, role_rows, name_hint):
    raw = open(path, "rb").read()
    L = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8") if l.strip()]
    rs = set(role_rows)
    bad = sorted({v["row"] for v in L if "row" in v and v["row"] not in rs})
    if bad:
        raise R.RowError(f"{path}: {len(bad)} rows are not in the expected row set (e.g. {bad[0]})")
    arms = EX.arms_of([v for v in L if "row" in v], rows)
    if len(arms) != 1:
        pick = [a for a in arms if name_hint.lower() in a.lower()]
        if len(pick) != 1:
            raise EX.ExportIncomplete(f"{path}: expected ONE arm (the pre-declared candidate {name_hint}), found {sorted(arms)}")
        L = [v for v in L if v.get("arm") == pick[0]]
        arms = {pick[0]: arms[pick[0]]}
    name, arm = next(iter(arms.items()))
    EX.classify(arm, full)
    it = [v.get("image_tokens") for v in L if v.get("image_tokens") is not None]
    it = [sum(x) if isinstance(x, list) else x for x in it]
    meta = dict(file=os.path.basename(path), path=os.path.abspath(path), sha256=hashlib.sha256(raw).hexdigest(), arm=name, lines=len(L), rows=len(arm),
                reads_per_row=[min(len(v) for v in arm.values()), max(len(v) for v in arm.values())],
                weights=sorted({json.dumps(v.get("weights"), sort_keys=True) for v in L if "weights" in v}),
                settings=sorted({json.dumps(v.get("settings"), sort_keys=True) for v in L if "settings" in v}),
                image_tokens=dict(min=min(it), mean=sum(it) / len(it), max=max(it)) if it else None,
                finish=dict(collections.Counter(v.get("finish") for v in L if "row" in v)))
    miss = [r for r in role_rows if r not in arm]
    if miss:
        raise EX.ExportIncomplete(f"{path}: {len(miss)} of {len(role_rows)} rows have no read (e.g. {miss[0]})")
    return arm, meta


def load_view_edit(ctx, key, role_rows):
    spec = ctx.cfg["exports"][key]
    return load_candidate(ctx.ex.path(key), ctx.rows, ctx.full, role_rows, spec["arm_hint"])


def references(ctx):
    floor, stock = ctx.ex.floor_and_stock(ctx.rows)
    EX.classify(floor, ctx.full)
    EX.classify(stock, ctx.full)
    for xs in floor.values():
        for x in xs:
            x["guess"] = {w: G.flag(dict(seed="guess", window=w, patterns=[]), x.get("tail")) for w in ctx.windows}
    return stock, floor


def judges(ctx, stock, floor, want_judge):
    rows, S, J = ctx.rows, ctx.S, ctx.J
    sp = P.tip_policies(max_vote=4, with_guess=False)
    fp = P.tip_policies(max_vote=5, with_guess=True, windows=ctx.windows)
    Ss = JV.Judge(rows, {r: stock[r] for r in S}, benches=B, policies=sp, roles=R.SIGNAL_ROLES, settings=ctx.settings)
    Fs = JV.Judge(rows, {r: floor[r] for r in S}, benches=B, policies=fp, roles=R.SIGNAL_ROLES, settings=ctx.settings)
    Sg = Fg = None
    if want_judge:
        Sg = JV.Judge(rows, {r: stock[r] for r in J}, benches=B, policies=sp, hull_from={p[2] for p in Ss.F.vertices}, hull_from_total={p[2] for p in Ss.Ft.vertices}, settings=ctx.settings)
        Fg = JV.Judge(rows, {r: floor[r] for r in J}, benches=B, policies=fp, hull_from={p[2] for p in Fs.F.vertices}, hull_from_total={p[2] for p in Fs.Ft.vertices}, settings=ctx.settings)
    return Ss, Fs, Sg, Fg


def point(Jx, lab, pol, arm):
    c = Jx.cand_from_views(f"{NAME} {lab}", pol, {NAME: arm}, kind="weights")
    n = len(Jx.J)
    gen, tot = sum(c["tokens"].values()) / n, sum(c["total"].values()) / n
    r = Jx.score(c)
    out = dict(budget=lab, acc=sum(c["value"].values()) / n, gen=gen, prompt=tot - gen, total=tot, multiple=gen / Jx.base_tokens, multiple_total=tot / Jx.base_total,
               judged=bool(r.get("judged")))
    if not r.get("judged"):
        out["error"] = r.get("error")
        return out, c, r
    e, t = r["eq"], r.get("total") or {}
    out.update(raw=r["raw"]["d"], d=e["d"], se=e["se"], p_macro=e["p_macro"], p_gen=e["p_gen"], carrier=e["carrier"], npos=e["npos"], neg=[b for b, _ in e["neg"]],
               p_bench_t=e["p_bench_t"], per_bench={b: e["per_bench"][b]["d"] for b in B}, per_bench_se={b: e["per_bench"][b]["se"] for b in B},
               ref=r["ref"], d_total=t.get("d"), se_total=t.get("se"), p_gen_total=t.get("p_gen"), ref_total=t.get("ref"))
    return out, c, r


def text(x):
    s = (f"  {x['budget']:28s} acc {100 * x['acc']:6.2f} | tokens generated {x['gen']:6.0f} + prompt {x['prompt']:6.0f} = total {x['total']:6.0f}"
         f" (x{x['multiple']:4.2f} generated, x{x['multiple_total']:4.2f} total of one reference read)")
    if not x["judged"]:
        return s + f" | NOT REFERENCED: {x.get('error')}"
    s += (f"\n  {'':28s} reference {JV.ref_text(x['ref'])} | raw vs 1 reference read {100 * x['raw']:+6.2f} | at equal generated tokens {100 * x['d']:+6.2f} +- {100 * x['se']:4.2f}"
          f" p_macro {x['p_macro']:.4f} p_gen {x['p_gen']:.4f} (worst without {x['carrier']}) positive {x['npos']}/{len(B)}"
          + (f" | NEGATIVE: {x['neg']}" if x["neg"] else "")
          + f"\n  {'':28s} per benchmark: " + " ".join(f"{b[:5]} {100 * x['per_bench'][b]:+5.1f}+-{100 * x['per_bench_se'][b]:3.1f}" for b in B))
    if x.get("d_total") is not None:
        s += f"\n  {'':28s} at equal TOTAL tokens (printed, not gating): {100 * x['d_total']:+6.2f} +- {100 * x['se_total']:4.2f} p_gen {x['p_gen_total']:.4f} | reference {JV.ref_text(x['ref_total'])}"
    return s


def base_line(Jx, name):
    V = {p[2]: p for p in Jx.F.points}
    one = Jx.F.table["vote1"]
    tot = sum(v["total"] for v in one.values()) / len(one)
    return (f"  {name}: one read {100 * V['vote1'][1]:.2f} (generated {Jx.base_tokens:.0f} + prompt {tot - Jx.base_tokens:.0f} = total {tot:.0f}); frontier vertices: "
            + " | ".join(f"{p[2]} {100 * p[1]:.2f} x{p[0] / Jx.F.base_tokens:.2f}" for p in Jx.F.vertices))


def compute(Jx, arm, Jb):
    n = min(len(arm[r]) for r in Jx.J)
    A = [point(Jx, lab, pol, {r: arm[r] for r in Jx.J}) for lab, pol in budgets(NAME, n)]
    Bi = [point(Jb, lab, pol, {r: arm[r] for r in Jb.J}) for lab, pol in budgets(NAME, n)]
    return A, Bi


def reads_per_row(arm):
    return max(len(v) for v in arm.values()) if arm else 0


def same_policy_line(arm, rows_):
    n = min(len(arm[r]) for r in rows_)
    return f"  the SAME policy on both systems (the candidate has {n} reads per row: its last point is a plain vote of {n}, the base's frontier there is the early-stopping vote of {n + 1}):"


def section(L, title, pts, Jx, name):
    L += ["", title, base_line(Jx, name)]
    for x, _, _ in pts:
        L.append(text(x))


def like_for_like(Jb, arm):
    n = min(len(arm[r]) for r in Jb.J)
    names = {"one read": "vote1", "default budget": "n2:disagree->tip", f"vote of {n}": f"vote{n}", "3 reads + 2 unless unanimous": "n3:disagree->tipx2"}
    out = []
    for lab, pol in budgets(NAME, n):
        fn = names.get(lab)
        if fn not in Jb.F.table:
            continue
        c = Jb.cand_from_views(f"{NAME} {lab}", pol, {NAME: {r: arm[r] for r in Jb.J}}, kind="weights")
        ft = Jb.F.table[fn]
        xd = {r: c["value"][r] - ft[r]["ok"] for r in Jb.J}
        blk = Jb._block(xd, B)
        m = len(Jb.J)
        cg, ct = sum(c["tokens"].values()) / m, sum(c["total"].values()) / m
        fg, ftot = sum(v["tokens"] for v in ft.values()) / m, sum(v["total"] for v in ft.values()) / m
        out.append(dict(budget=lab, policy=fn, acc=sum(c["value"].values()) / m, acc_floor=sum(v["ok"] for v in ft.values()) / m, d=blk["d"], se=blk["se"],
                        p_two_sided=Jb.p_two_sided(xd, B), npos=blk["npos"], per_bench={b: blk["per_bench"][b]["d"] for b in B},
                        gen_ratio=cg / fg, prompt_ratio=(ct - cg) / (ftot - fg), total_ratio=ct / ftot, gen=cg, prompt=ct - cg, total=ct, floor_gen=fg, floor_prompt=ftot - fg, floor_total=ftot))
    return out


def lfl_text(x):
    return (f"  {x['budget']:28s} weights {100 * x['acc']:6.2f} vs request floor {100 * x['acc_floor']:6.2f}: {100 * x['d']:+5.2f} +- {100 * x['se']:4.2f} (two-sided p {x['p_two_sided']:.3f}; positive {x['npos']}/{len(B)})"
            f" | tokens weights / floor: generated {x['gen']:.0f} / {x['floor_gen']:.0f} = x{x['gen_ratio']:.2f}, prompt {x['prompt']:.0f} / {x['floor_prompt']:.0f} = x{x['prompt_ratio']:.2f},"
            f" total {x['total']:.0f} / {x['floor_total']:.0f} = x{x['total_ratio']:.2f} | " + " ".join(f"{b[:5]} {100 * x['per_bench'][b]:+5.1f}" for b in B))


def run_signal(ctx):
    os.makedirs(ctx.out_dir, exist_ok=True)
    t0 = time.time()
    arm, meta = load_view_edit(ctx, "view_edit_signal", ctx.S)
    stock, floor = references(ctx)
    Ss, Fs, _, _ = judges(ctx, stock, floor, False)
    A, Bi = compute(Ss, arm, Fs)
    L = [f"{NAME} on the SIGNAL rows ({len(ctx.S)} SHOW + MONITOR rows): the signal computation of the verdict; never a decision",
         f"candidate reads: {json.dumps(meta)}"]
    section(L, "A. against the STOCK model's own frontier (frontier against frontier at equal generated tokens)", A, Ss, f"STOCK ({reads_per_row(stock)} cached reads)")
    d = A[1][0]
    ok = bool(d["judged"] and d["d"] > 0 and d["p_macro"] <= ctx.prescreen_p)
    L.append(f"  -> pre-screen rule (default budget: gain > 0 with p_macro <= {ctx.prescreen_p}): {'PASSES' if ok else 'does NOT pass'} (the candidate is pre-declared; its JUDGE reads are read whatever this line says)")
    section(L, "B. INFORMATION: against the request-floor base (floor-only tip; decides nothing)", Bi, Fs, f"request-floor base ({reads_per_row(floor)} cached reads)")
    lfl = like_for_like(Fs, arm)
    L += [same_policy_line(arm, Fs.J)] + [lfl_text(x) for x in lfl]
    L.append(f"\n({time.time() - t0:.0f} s)")
    open(os.path.join(ctx.out_dir, "view_edit_signal.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    path = os.path.join(ctx.out_dir, "view_edit_signal.json")
    JV.dump(dict(meta=meta, A=[x for x, _, _ in A], B=[x for x, _, _ in Bi], B_same_policy=lfl, prescreen_pass=ok), path)
    return dict(signal=path, prescreen_pass=ok)


def run_judge(ctx):
    t0 = time.time()
    arm, meta = load_view_edit(ctx, "view_edit_judge", ctx.J)
    sig = json.load(open(os.path.join(ctx.out_dir, "view_edit_signal.json"), encoding="utf-8"))
    stock, floor = references(ctx)
    Ss, Fs, Sg, Fg = judges(ctx, stock, floor, True)
    A, Bi = compute(Sg, arm, Fg)
    rec = Sg.decide([A[1][1]], family=f"{NAME}, pre-declared, K = 1")[0]
    acc = bool(rec["accept"])
    eqd = rec.get("eq") or {}
    L = [f"VERDICT for {NAME} (a weight change: vision patch-embedding kernel resampled to a finer patch + its preprocessor bounds; STOCK requests)",
         f"{len(ctx.J)} JUDGE rows; candidate reads: {json.dumps(meta)}", *ctx.ex.reads_lines(), "",
         f"VERDICT A: {'ACCEPTED' if acc else 'NOT ACCEPTED'} on the default budget: {rec['reason']}"]
    section(L, f"A. against the STOCK model's own frontier (frontier against frontier at equal generated tokens; all {len(B)} benchmarks scored; K = 1)", A, Sg, f"STOCK ({reads_per_row(stock)} cached reads)")
    sd, jd = sig["A"][1], A[1][0]
    kept = jd["d"] / sd["d"] if (sd.get("judged") and jd.get("judged") and sd["d"] > 0) else None
    L += ["", "SHRINKAGE for A (default budget, equal generated tokens): signal rows "
          + (f"{100 * sd['d']:+.2f} +- {100 * sd['se']:.2f}" if sd.get("judged") else "n/a") + " -> JUDGE rows "
          + (f"{100 * jd['d']:+.2f} +- {100 * jd['se']:.2f}" if jd.get("judged") else "n/a") + (f" (kept {100 * kept:.0f}%)" if kept is not None else "")
          + "; one read: signal " + (f"{100 * sig['A'][0]['d']:+.2f}" if sig["A"][0].get("judged") else "n/a") + " -> JUDGE " + (f"{100 * A[0][0]['d']:+.2f}" if A[0][0].get("judged") else "n/a")]
    section(L, "B. INFORMATION (decides nothing): against the request-floor base, frontier against frontier", Bi, Fg, f"request-floor base ({reads_per_row(floor)} cached reads)")
    lfl = like_for_like(Fg, arm)
    L += [same_policy_line(arm, Fg.J)] + [lfl_text(x) for x in lfl]
    if lfl:
        x1 = lfl[0]
        L.append(f"  -> one read, weights minus request floor: {100 * x1['d']:+.2f} +- {100 * x1['se']:.2f} pp ("
                 + ("within noise of the request floor" if x1["p_two_sided"] > 0.10 else "NOT within noise of the request floor")
                 + f", two-sided p {x1['p_two_sided']:.3f}) at x{x1['total_ratio']:.2f} total tokens (generated x{x1['gen_ratio']:.2f}, prompt x{x1['prompt_ratio']:.2f})")
    out = dict(meta=meta, export_reads=ctx.ex.reads, judged=bool(rec.get("judged")), accepted=acc, reason=rec["reason"], A=[x for x, _, _ in A], B=[x for x, _, _ in Bi], B_same_policy=lfl, signal=sig, shrinkage=dict(signal=sd.get("d"), judge=jd.get("d"), kept=kept),
               decision=dict(p_gen=eqd.get("p_gen"), p_macro=eqd.get("p_macro"), d=eqd.get("d"), se=eqd.get("se"), K=1, alpha=ctx.settings["alpha"], neg=[b for b, _ in eqd.get("neg", [])]))
    if acc:
        out["tip_file"] = write_tip(ctx, arm, meta, rec, A, sig)
        L += ["", f"TIP FILE of the weights system: {out['tip_file']}"]
    else:
        old = os.path.join(ctx.out_dir, TIP_FILE)
        if os.path.exists(old):
            moved = R.superseded_name(old)
            os.replace(old, moved)
            L += ["", f"TIP FILE: none (not accepted); the {TIP_FILE} of an earlier run was moved to {os.path.basename(moved)}"]
    L.append(f"\n({time.time() - t0:.0f} s)")
    open(os.path.join(ctx.out_dir, "view_edit_judge.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    path = os.path.join(ctx.out_dir, "view_edit_judge.json")
    JV.dump(out, path)
    return dict(judge=path, accepted=acc, tip_file=out.get("tip_file"))


def write_tip(ctx, arm_j, meta, rec, A, sig):
    rows = ctx.rows
    arm_s, _ = load_candidate(sig["meta"]["path"], rows, ctx.full, ctx.S, ctx.cfg["exports"]["view_edit_signal"]["arm_hint"])
    n = min(min(len(v) for v in arm_s.values()), min(len(v) for v in arm_j.values()))
    pols = P.tip_policies(max_vote=min(5, n), with_guess=False)
    Ws = JV.Judge(rows, arm_s, benches=B, policies=pols, roles=R.SIGNAL_ROLES, settings=ctx.settings)
    Wg = JV.Judge(rows, arm_j, benches=B, policies=pols, hull_from={p[2] for p in Ws.F.vertices}, hull_from_total={p[2] for p in Ws.Ft.vertices}, settings=ctx.settings)
    V = [p[2] for p in sorted(Ws.F.vertices)]
    pts = []
    for nm in sorted(Wg.F.table, key=lambda k: P.macro_mean(Wg.F.table[k], rows, "tokens")):
        tj, ts = Wg.F.table[nm], Ws.F.table[nm]
        pts.append(dict(name=nm, vertex=nm in V, multiple=P.macro_mean(tj, rows, "tokens") / Wg.F.base_tokens, multiple_total=P.macro_mean(tj, rows, "total") / Wg.Ft.base_tokens,
                        acc_judge=P.macro_mean(tj, rows, "ok"), acc_signal=P.macro_mean(ts, rows, "ok")))
    default = ctx.default_policy if ctx.default_policy in Wg.F.table else "vote1"
    w = [json.loads(x) for x in meta["weights"]]
    tip = dict(id="view_edit", parent="stock", kind="weights system",
               note=(f"General layer = the View Edit weights with STOCK requests. The request floor ({EX.settings_text(ctx.ex.base()['settings'])}) is NOT in this system; "
                     "the specialisation layer is empty."),
               weights=dict(name=NAME, artefact=w[0] if len(w) == 1 else w, description="vision patch-embedding kernel resampled to a finer patch (kernel' = B^T kernel) and its preprocessor bounds; "
                            "patch size and image-token bounds are in the View Edit report", requests="stock requests: no min_image_tokens floor", image_tokens=meta.get("image_tokens")),
               entries=[dict(condition="always", cond="always", procedure=None, settings={}, ops=[], weights=w[0] if len(w) == 1 else w, cid="view_edit", author="pre-declared",
                             source=[], audit="GENERAL",
                             verdict=dict(layer="GENERAL", basis="verdict A: against the STOCK model's own frontier, default budget, K = 1", d=rec["eq"]["d"], se=rec["eq"]["se"],
                                          p_gen=rec["eq"]["p_gen"], scored=list(B), benches=list(B), one_read_raw=A[0][0].get("raw")))],
               special=[],
               policy=dict(default=default, extras=[], guess=None,
                           frontier=dict(vertices=V, points=pts, default=default, report=["vote1", default, V[-1]], base_tokens=Wg.F.base_tokens, base_total=Wg.Ft.base_tokens,
                                         reads_per_row=n, note=f"from the candidate's own {n} reads per row; vertices chosen on signal rows, valued on JUDGE rows")),
               reads=dict(signal=sig["meta"], judge=meta))
    p = os.path.join(ctx.out_dir, TIP_FILE)
    JV.dump(tip, p)
    return p
