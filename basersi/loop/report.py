import collections
import json
import os
import time


def report_path(run_dir, generation):
    return os.path.join(run_dir, f"report_g{generation:03d}.txt")


def ledger_path(run_dir):
    return os.path.join(run_dir, "ledger.jsonl")


def ledger_key(c):
    return json.dumps([c.get("gen"), c.get("cid")])


def written_keys(path, key):
    out = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.add(key(json.loads(line)))
                except ValueError:
                    continue
    return out


def lines(run):
    G = run.G()
    tip0 = G.get("tip_at_start")
    L = [f"BaseRSI loop, generation {run.g}: tip at start {tip0} -> tip now {run.tip()['id']}", ""]
    fr = G.get("frontier") or {}
    L.append(f"READ POLICY of the tip at start (frontier vertices chosen on signal rows, valued on JUDGE rows; default = {fr.get('default')}; the budget is a reporting choice)")
    for p in fr.get("points", []):
        if p["name"] in fr.get("report", []) or p["vertex"]:
            L.append(f"  {p['name']:30s} JUDGE acc {100 * p['acc_judge']:6.2f} (signal {100 * p['acc_signal']:6.2f}) tokens x{p['multiple']:4.2f} (total x{p['multiple_total']:4.2f})"
                     + ("  <- default budget" if p["name"] == fr.get("default") else "") + ("  [vertex]" if p["vertex"] else ""))
    L.append("")
    L.append("SIGNALS (SHOW + MONITOR rows)")
    for b, s in (G.get("signals") or {}).get("by_bench", {}).items():
        L.append(f"  {b:12s} acc {100 * s['acc']:5.1f} disagreement of two reads {100 * s['disagree']:4.1f}% declared guess {100 * s['guess'][str(run.windows[-1])]:4.1f}% tokens {s['tokens']:6.0f}")
    L += ["", "CANDIDATES"]
    for c in G.get("cands", []):
        s = c.get("screen") or {}
        L.append(f"- {c['cid']} {c['kind']:8s} {c['name']} | source {c.get('source') or '-'}" + ("" if c.get("line") else f" | {c.get('status')}"))
        if c.get("skipped_rows"):
            L.append(f"    rows the candidate could not be read on (error / op_skipped lines; they count as the tip): {c['skipped_rows']}")
        if c.get("why"):
            L.append(f"    proposer: {str(c['why'])[:200]!r}" + (f" | {json.dumps({k_: c['entry'].get(k_) for k_ in ('condition', 'procedure', 'settings') if c['entry'].get(k_)}, ensure_ascii=False)[:300]}" if c.get("entry") else ""))
        if s.get("judged"):
            L.append(f"    signal rows: x{s['multiple']:.2f} raw {100 * s['raw']:+.2f} at equal compute {100 * s['d']:+.2f} +- {100 * s['se']:.2f} p_macro {s['p_macro']:.3f}")
        for lab_, key_ in (("signal rows", "budgets_signal"), ("JUDGE rows ", "budgets_judge")):
            if c.get(key_):
                L.append(f"    {lab_}, frontier against frontier: " + " | ".join(
                    f"{b_} acc {100 * v_['acc']:.2f} (x{v_['multiple']:.2f}) vs the tip at equal tokens {100 * v_['d']:+.2f} +- {100 * v_['se']:.2f} p_gen {v_['p_gen']:.3f}"
                    for b_, v_ in c[key_].items() if v_.get("judged")))
        if c.get("line"):
            L += ["    " + x for x in c["line"].split("\n")]
    if G.get("cards") or G.get("dropped"):
        L += ["", f"PROPOSER: cards {G.get('cards')}; dropped proposals ({len(G.get('dropped') or [])}), reason and first 200 characters:"]
        for d in G.get("dropped") or []:
            L.append(f"  - [{'mixed card' if d.get('mixed') else 'card of ' + ','.join(d.get('source') or [])}] {d['why']} | {d['head']!r}")
    L += ["", "SIGNAL-ROW GAIN AGAINST JUDGE-ROW GAIN (equal compute; how much of a signal-row gain is fit)"]
    for r in run.state["shrinkage"][-20:]:
        if r.get("kind") == "admission":
            L.append(f"  g{r['gen']} admission {r['name'][:38]:38s} source {','.join(r['source'])}: JUDGE rows {100 * r['source_judge_d']:+5.2f} +- {100 * r['source_judge_se']:4.2f}"
                     + (f" | MONITOR rows {100 * r['source_monitor_d']:+5.2f} +- {100 * r['source_monitor_se']:4.2f}" if r.get("source_monitor_d") is not None else " | MONITOR rows: none")
                     + (" | ADMITTED (ID_SPECIFIC)" if r.get("admitted") else " | not admitted"))
            continue
        L.append(f"  g{r['gen']} {r['kind']:8s} {r['name'][:40]:40s} signal {100 * r['signal_d']:+5.2f} +- {100 * r['signal_se']:4.2f} | JUDGE {100 * r['judge_d']:+5.2f} +- {100 * r['judge_se']:4.2f}"
                 + (f" | kept {100 * r['kept']:.0f}%" if r.get("kept") is not None else ""))
    t = run.tip()
    L += ["", f"TIP {t['id']}",
          "  GENERAL system: " + "; ".join(f"{e['cid']} ({'settings ' + json.dumps(e['settings']) if e.get('settings') else 'operator' if e.get('ops') else 'entry'})"
                                          for e in t["entries"]) + f" + read policy {t['policy'].get('default')}" + "".join(f" + {x['name']}" for x in t["policy"].get("extras", []))]
    L.append(f"  SPECIALISATION layer (OFF in the general system; never counted as a general gain): {len(t['special'])} entries")
    for e in t["special"]:
        v = e.get("verdict") or {}
        L.append(f"    {e['cid']} source {e.get('source') or '?'}: {v.get('layer')}"
                 + (f", off-source {100 * v['d']:+.2f} pp" + (f" (p_gen {v['p_gen']:.3f})" if v.get("p_gen") is not None else "") if v.get("d") is not None else "")
                 + (f", all benchmarks {100 * v['pooled']:+.2f} pp" if v.get("pooled") is not None else "")
                 + (f", source benchmark JUDGE rows {100 * v['source_d']:+.2f} +- {100 * v['source_se']:.2f} pp (p {v['source_p']:.4f}, K {v.get('K')})" if v.get("source_d") is not None else "")
                 + (f", source benchmark MONITOR rows {100 * v['monitor_d']:+.2f} +- {100 * v['monitor_se']:.2f} pp" if v.get("monitor_d") is not None else ""))
    if G.get("tip_change"):
        L.append(f"  changes this generation: {'; '.join(G['tip_change'])}")
        L.append(f"  cache of the new tip: {G.get('new_tip_cache')}")
    both = both_systems(run)
    if both:
        L += ["", "BOTH SYSTEMS on JUDGE rows (one read; the specialisation system is reported, not optimised)"] + both
    return L


def both_systems(run):
    t, out = run.tip(), []
    run.load_routes(t["entries"] + t["special"], run.JR)
    both = sorted(t["entries"] + t["special"], key=lambda e: e.get("pos", 10 ** 6))
    for name, ents in (("general", t["entries"]), ("general + specialisation", both)):
        arm = run.arm(ents, run.JR)
        n = min([len(arm.get(r, ())) for r in run.JR])
        if n < 2:
            out.append(f"  {name:26s} not read (fewer than 2 cached reads on some JUDGE row)")
            continue
        byb = collections.defaultdict(list)
        for r in run.JR:
            byb[run.rows[r]["bench"]].append(sum(x["ok"] for x in arm[r]) / len(arm[r]))
        acc = sum(sum(v) / len(v) for v in byb.values()) / len(byb)
        out.append(f"  {name:26s} acc {100 * acc:6.2f} (>= {n} reads per row) | " + " ".join(f"{b[:5]} {100 * sum(v) / len(v):5.1f}" for b, v in sorted(byb.items())))
    return out if t["special"] else out[:1]


def ledger_lines(run, now=None):
    t = time.time() if now is None else now
    return [json.dumps(dict({k: v for k, v in c.items() if k not in ("line", "card")}, t=t), ensure_ascii=False) for c in run.G().get("cands", [])]


def write(run):
    p = report_path(run.rd, run.g)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(lines(run)) + "\n")
    os.replace(p + ".tmp", p)
    lp = ledger_path(run.rd)
    have = written_keys(lp, ledger_key)
    with open(lp, "a", encoding="utf-8") as fh:
        for line in ledger_lines(run):
            if ledger_key(json.loads(line)) not in have:
                fh.write(line + "\n")
    run.log(f"  {os.path.basename(p)} written")
    return p
