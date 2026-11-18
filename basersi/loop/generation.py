import collections
import hashlib
import json
import math
import os
import sys
import time

from basersi import config
from basersi import fingerprint as FP
from basersi.data.rows import PROPOSER_ROLES, ROLE_FIELD, SIGNAL_ROLES, VALIDATION_ROLE
from basersi.engine import dispatch, grader, reads, router, serve
from basersi.engine import request as RQ
from basersi.judge import frontier as F
from basersi.judge import guess as GU
from basersi.judge import judge as JV
from basersi.judge import reads as JR
from basersi.judge import stats as ST
from basersi.loop import proposer as PR
from basersi.loop import report
from basersi.loop.queue_client import QueueClient

STEPS = ["signals", "candidates", "prescreen", "judge", "update", "report"]
RETEST = ("all", "not_covered_by_verdict", "none")
ORIGIN_FREE_KEYS = ("wait_seconds", "poll_seconds", "priorities", "judge_export")
MONITOR_ROLES = tuple(r for r in SIGNAL_ROLES if r not in PROPOSER_ROLES)
BASE_ENTRY = dict(condition="always", cond="always", procedure=None, settings={}, ops=[])


class Refuse(config.Refused):
    pass


def sha16(x):
    return hashlib.sha256(x.encode()).hexdigest()[:16]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(p, x):
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(x, f, ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


def row_key(row):
    return json.dumps([row.get("gen"), row.get("cid"), row.get("kind")])


def tip_id(generation):
    return f"tip{generation:03d}"


def init_state(start_tip, sources=None, verdicts=None, default_policy=None, note=""):
    default_policy = default_policy or config.load("loop")["default_read_policy"]
    ents, special = [], []
    for e in start_tip.get("entries") or []:
        cid = e.get("from_cand") or e.get("cid")
        x = dict(condition=e["condition"], cond="always" if e["condition"] == "always" else sha16(e["condition"]), procedure=e.get("procedure"),
                 settings=e.get("settings") or {}, ops=e.get("ops") or [], cid=cid, author=e.get("author"), source=list((sources or {}).get(cid) or []),
                 pos=len(ents) + len(special))
        if e.get("edit"):
            raise Refuse(f"start tip entry {cid} carries a weight edit: the loop proposes and reads harness entries only")
        if x["cond"] == "always":
            ents.append(x)
            continue
        v = (verdicts or {}).get(cid)
        if v is None:
            ents.append(dict(x, audit="pending"))
        elif v.get("layer") == "GENERAL":
            ents.append(dict(x, audit="GENERAL", verdict=v))
        else:
            special.append(dict(x, verdict=v))
    return dict(created=time.time(), gen=0, gens={}, tips=[], shrinkage=[], note=note or f"from start tip {start_tip.get('id')}",
                tip=dict(id=tip_id(0), born=0, parent=start_tip.get("id"), entries=ents, special=special,
                         policy=dict(default=default_policy, extras=[], frontier=None, guess=None)))


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise config.MissingInput(f"{path} does not exist") from None
    except ValueError as e:
        raise config.Refused(f"{path} is not a json file: {e}") from None


def verdicts_from_audit(audit_path, members_path):
    members = _read_json(members_path)
    audit = _read_json(audit_path)
    if "entry_audit" not in audit:
        raise Refuse(f"{audit_path} has no entry_audit records: pass an acceptance-test file, or a json {{entry id: verdict}} without --members")
    out = {}
    for r in audit["entry_audit"]:
        if r.get("name") not in members:
            raise Refuse(f"{members_path} does not name the entries of the acceptance-test record {r.get('name')!r} of {audit_path}")
        for cid in members[r["name"]]:
            eq, pooled = r.get("eq") or {}, r.get("pooled") or {}
            out[cid] = dict(layer=r["layer"], d=eq.get("d"), se=eq.get("se"), p_gen=eq.get("p_gen"), pooled=pooled.get("d"), pooled_p=pooled.get("p_macro"),
                            scored=r.get("scored") or [], benches=sorted(set(r.get("scored") or []) | set(r.get("source") or [])), audit=r["name"], gen=0,
                            basis="acceptance-test verdicts of the start tip")
    return out


class LoopRun:
    def __init__(self, run_dir, rows_path, queue_dir, expected_model, loop_config=None, offline=False, logger=log,
                 judge_settings=None, accept_other_code=False):
        self.cfg = loop_config or config.load("loop")
        self.judge_settings = dict(judge_settings or JV.SHIPPED)
        c = self.cfg
        self.tip_reads, self.min_tip_reads = int(c["tip_reads_per_row"]), int(c["min_tip_reads_per_row"])
        self.s_screen, self.s_judge, self.s_audit = int(c["screen_samples"]), int(c["judge_samples"]), int(c["audit_samples"])
        self.screen_p, self.k_max = float(c["screen_p_macro"]), int(c["max_sent_to_judge"])
        self.windows = tuple(int(w) for w in c["guess_tail_windows"])
        if max(self.windows) > int(c["kept_tail_chars"]):
            raise Refuse(f"guess_tail_windows {list(self.windows)} reach beyond kept_tail_chars {c['kept_tail_chars']}: the longer windows would see a cut tail")
        self.max_vote, self.arm_min_reads = int(c["max_vote_reads"]), int(c["settings_arm_min_reads"])
        self.default_policy = c["default_read_policy"]
        self.retest = c["retest_specialisation"]
        if self.retest not in RETEST:
            raise Refuse(f"retest_specialisation {self.retest!r}: one of {', '.join(RETEST)}")
        self.prop = c["proposer"]
        PR.configure(self.prop)
        self.prio = c["priorities"]
        self.wait_s, self.poll_s = float(c["wait_seconds"]), float(c["poll_seconds"])
        self.offline = bool(offline)
        self.log = logger
        self.rd = run_dir
        try:
            checked = reads.verified_rows(rows_path)
            self.development_rows = JR.load_rows(rows_path)
            self.rows = PR.load_rows(rows_path)
        except config.Refused as ex:
            raise Refuse(str(ex)) from None
        try:
            model = dispatch.expected_record(expected_model)
        except serve.ServerMismatch as ex:
            raise Refuse(str(ex))
        self.menu = c["settings_menu"]
        self.stock = {k: self.menu[k]["stock"] for k in self.menu}
        self.role = collections.defaultdict(list)
        for i, r in self.rows.items():
            self.role[r[ROLE_FIELD]].append(i)
        for k in self.role:
            self.role[k].sort()
        sp = os.path.join(self.rd, "state.json")
        self.state = _read_json(sp) if os.path.exists(sp) else init_state(dict(id="stock", entries=[]), default_policy=self.default_policy,
                                                                          note="from the stock model")
        self.origin = dict(queue_dir=os.path.realpath(queue_dir), rows_sha256=FP.file_sha256(rows_path),
                           rows_manifest_sha256=config.rows_manifest_sha256(),
                           loop_config_sha256=FP.digest(FP.canonical({k: v for k, v in c.items() if k not in ORIGIN_FREE_KEYS})),
                           read_settings_sha256=reads.read_settings_sha256(), judge_settings_sha256=FP.digest(FP.canonical(self.judge_settings)), model=model,
                           code=config.code_identity())
        have = self.state.get("origin")
        if have is None and self.state["gens"]:
            raise Refuse(f"{sp} records no origin (queue, inputs) and has generations: it is not continued; use a new run folder (this one is kept)")
        diff = sorted(k for k in set(have or {}) | set(self.origin) if (have or {}).get(k) != self.origin.get(k)) if have is not None else []
        if diff == ["code"] and accept_other_code:
            print(f"[basersi] continuing {sp}: its generations were made by other package code (version {have['code'].get('version')}, "
                  f"sources {str(have['code'].get('sources_sha256'))[:12]}) with the same inputs, because {FP.OTHER_CODE_KEY} is true",
                  file=sys.stderr, flush=True)
            reused = self.state.setdefault("reused_with_other_code", [])
            if not any(x.get("recorded") == have["code"] and x.get("now") == self.origin["code"] for x in reused):
                reused.append(dict(recorded=have["code"], now=self.origin["code"], key=FP.OTHER_CODE_KEY, at=time.strftime("%Y-%m-%d %H:%M:%S %z")))
        elif diff == ["code"]:
            raise Refuse(f"{sp} was made by other package code with the same inputs: set {FP.OTHER_CODE_KEY} to continue it knowingly "
                         "(the reuse is recorded in state.json and printed), or use a new run folder; this one is kept")
        elif diff:
            raise Refuse(f"{sp} belongs to a run with other inputs ({', '.join(diff)} differ): "
                         "a run folder is never continued with other inputs. Use a new run folder (another run_name, or --run-dir); this one is kept")
        try:
            dispatch.check_queue_settings(queue_dir, record=False)
            dispatch.expect_model(queue_dir, expected_model, record=False)
        except serve.ServerMismatch as ex:
            raise Refuse(str(ex))
        if not self.offline:
            try:
                RQ.verify_images(checked.values())
            except RQ.ImageRefused as ex:
                raise Refuse(f"{rows_path}: {ex}")
        dispatch.check_queue_settings(queue_dir)
        dispatch.expect_model(queue_dir, expected_model)
        self.io = QueueClient(queue_dir, log=self.log, kept_tail_chars=int(c["kept_tail_chars"]), offline=self.offline)
        os.makedirs(self.rd, exist_ok=True)
        self.state["origin"] = self.origin
        self.save()
        self.routes, self.route_p = {}, {}
        self.run_name = os.path.basename(os.path.normpath(self.rd))
        self.benches = sorted({r["bench"] for r in self.rows.values()})
        self.SIGR = [i for role in SIGNAL_ROLES for i in self.role[role]]
        self.JR = list(self.role[VALIDATION_ROLE])
        self._jg, self._jsp = None, None
        self.g = self.state.get("gen", 0)

    def save(self):
        jdump(os.path.join(self.rd, "state.json"), self.state)

    def tip(self):
        return self.state["tip"]

    def wait(self, keys, label):
        return self.io.wait(keys, time.time() + self.wait_s, self.poll_s, label)

    def settings_of(self, entries):
        s = dict(self.stock)
        for e in entries:
            s.update(e.get("settings") or {})
        return s

    def sig(self, prefix, settings, ops=None):
        s = {k: v for k, v in sorted(settings.items()) if v != self.stock.get(k)}
        return sha16(json.dumps([prefix or None, s] + ([PR.canon_ops(ops)] if ops else []), sort_keys=True))

    def fires(self, e, r):
        return e["cond"] == "always" or bool(self.routes.get((e["cond"], r)))

    def view(self, entries, r):
        f = [e for e in entries if self.fires(e, r)]
        pre = "\n\n".join(e["procedure"] for e in f if e.get("procedure")) or None
        st = self.settings_of(f)
        return pre, st, self.sig(pre, st, self.ops_of(f))

    @staticmethod
    def ops_of(fired):
        ls = [e.get("ops") for e in fired if e.get("ops")]
        return PR.merge_ops(ls) if ls else []

    @property
    def stock_sig(self):
        if not hasattr(self, "_stock_sig"):
            self._stock_sig = self.sig(None, dict(self.stock))
        return self._stock_sig

    def read_key(self, sg, r, s, salt=None):
        return (salt or "") + (f"p000|parent|{r}|{s}" if sg == self.stock_sig else f"x|{sg}|{r}|{s}")

    def key_salt(self, c):
        fmt = self.cfg.get("audit_key_salt")
        return fmt.format(cid=c["cid"]) if fmt and c.get("kind") == "audit" else None

    def read_req(self, entries, r, s, prio, salt=None):
        pre, st, sg = self.view(entries, r)
        req_st = {k: (None if st[k] == self.stock[k] else st[k]) for k in self.menu}
        q = dict(key=self.read_key(sg, r, s, salt), mode="read", priority=prio, row=r, sample=s, prefix=pre, settings=req_st)
        ops = self.ops_of([e for e in entries if self.fires(e, r)])
        if ops:
            q["ops"] = ops
        return q

    def reqs(self, entries, rows, samples, prio=None, salt=None):
        prio = self.prio["candidate"] if prio is None else prio
        return [self.read_req(entries, r, s, prio, salt) for r in rows for s in samples]

    def samples(self, entries, r, upto=None):
        _, _, sg = self.view(entries, r)
        out = []
        for s in range(self.tip_reads if upto is None else upto):
            x = self.io.get(self.read_key(sg, r, s))
            if x is not None:
                out.append(x)
        return out

    def route_reqs(self, e, rows):
        if e["cond"] == "always":
            return []
        return [dict(key=router.route_key(e["cond"], r), mode="route", priority=self.prio["route"], row=r, condition=e["condition"], view=None) for r in rows]

    def load_routes(self, entries, rows):
        for e in entries:
            if e["cond"] == "always":
                continue
            for r in rows:
                x = self.io.get(router.route_key(e["cond"], r))
                if x is not None:
                    self.routes[(e["cond"], r)] = router.routed(x)
                    self.route_p[(e["cond"], r)] = x.get("p")

    def ensure_routes(self, entries, rows, label):
        reqs = [q for e in entries for q in self.route_reqs(e, rows)]
        if reqs:
            n = self.io.submit(f"g{self.g:03d}_route_{label}", reqs)
            done, miss = self.wait([q["key"] for q in reqs], f"routes {label}")
            if miss:
                self.log(f"  routes {label}: {len(miss)} unanswered at the deadline (treated as no)")
            errs = sum(1 for q in reqs if q["key"] in self.io.err and not self.io.have(q["key"]))
            if errs:
                self.log(f"  routes {label}: {errs} answered with an error (treated as no)")
            self.log(f"  routes {label}: {len(reqs)} asked, {n} new")
        self.load_routes(entries, rows)

    def routes_for(self, entries, rows, label):
        if self.offline:
            self.load_routes(entries, rows)
        else:
            self.ensure_routes(entries, rows, label)

    def tip_view(self):
        return {"settings": {}, "entries": [dict(condition=e["condition"], procedure=e.get("procedure"), settings=e.get("settings") or {}, ops=e.get("ops") or [])
                                            for e in self.tip()["entries"]]}

    def residual(self, parent_show):
        res = collections.defaultdict(float)
        for r, vs in parent_show.items():
            vs = [s for s in vs if s.get("ok") is not None]
            if vs:
                res[self.rows[r]["bench"]] += 1.0 - sum(bool(s["ok"]) for s in vs) / len(vs)
        return dict(res)

    @staticmethod
    def alloc_sources(n, resid):
        src = sorted(b for b, v in resid.items() if v > 0)
        if n <= 0 or not src:
            return {}
        tot = sum(resid[b] for b in src)
        q = {b: n * resid[b] / tot for b in src}
        a = {b: int(math.floor(q[b])) for b in src}
        for b in sorted(src, key=lambda b: (-(q[b] - a[b]), b))[: n - sum(a.values())]:
            a[b] += 1
        if n >= len(src):
            for b in sorted((b for b in src if a[b] == 0), key=lambda b: (-resid[b], b)):
                donor = max((d for d in src if a[d] >= 2), key=lambda d: (a[d] - q[d], d))
                a[donor] -= 1
                a[b] += 1
        return {b: a[b] for b in src if a[b] > 0}

    def read(self, name, reqs, label=None):
        keys = list(dict.fromkeys(q["key"] for q in reqs))
        if not keys:
            return 0
        if self.offline:
            self.io.poll()
            miss = [k for k in keys if not self.io.have(k)]
            if miss:
                self.log(f"  {name}: OFFLINE, {len(miss)} of {len(keys)} reads are not recorded")
            return len(miss)
        n = self.io.submit(name, reqs)
        done, miss = self.wait(keys, label or name)
        over = [k for k in keys if k not in self.io.res and (self.io.err.get(k) or {}).get("image_limit")]
        self.log(f"  {name}: {len(keys)} reads, {n} new, {len(miss)} unanswered" + (f", {len(over)} above the server image limit (missing)" if over else ""))
        return len(miss) + len(over)

    def arm(self, entries, rows, upto=None, tail=False, salt=None):
        upto = self.tip_reads if upto is None else upto
        out = {}
        for r in rows:
            _, _, sg = self.view(entries, r)
            row, xs = self.rows[r], []
            for s in range(upto):
                x = self.io.get(self.read_key(sg, r, s, salt))
                if x is None or x.get("op_skipped") or (x.get("ok") is None and x.get("finish") is None):
                    continue
                ok, p = grader.graded_record(x, row)
                c = grader.answer_class(p, row)
                d = dict(ok=bool(ok), tokens=float(x.get("tokens") or 0.0), prompt_tokens=float(x.get("prompt_tokens") or 0.0), pred=p,
                         cls=F.NONE if c == grader.NO_ANSWER_CLASS else c, finish=x.get("finish"), s=s)
                if tail:
                    d["tail"] = x.get("reasoning") or ""
                xs.append(d)
            if xs:
                out[r] = xs
        return out

    def fired_rows(self, e, rows):
        return [r for r in rows if self.fires(e, r)]

    def judges(self):
        if self._jg is not None:
            return self._jg
        tip = self.tip()["entries"]
        tS, tJ = self.arm(tip, self.SIGR, tail=True), self.arm(tip, self.JR, tail=True)
        nmin = min([len(tS.get(r, ())) for r in self.SIGR] + [len(tJ.get(r, ())) for r in self.JR])
        if nmin < self.min_tip_reads:
            raise Refuse(f"the tip {self.tip()['id']} has {nmin} cached reads on some development row (< {self.min_tip_reads}): fill its cache first")
        base = [dict(p) for p in self.cfg["signal_frontier_policies"]]
        F0 = F.Frontier(tS, self.SIGR, self.development_rows, policies=base)
        V = F0.vertices
        price = (V[1][1] - V[0][1]) / ((V[1][0] - V[0][0]) / V[0][0]) if len(V) > 1 else 0.0
        det = GU.derive([dict(row=r, ok=x["ok"], tokens=x["tokens"], tail=x["tail"]) for r in self.SIGR for x in tS[r]], self.development_rows, price, windows=self.windows)
        for armx, cross in ((tS, True), (tJ, False)):
            for r, xs in armx.items():
                pats = (det.get("picked") or {}).get(self.rows[r]["bench"], det["patterns"]) if cross else det["patterns"]
                for x in xs:
                    x["guess"] = {w: GU.flag(dict(seed=det["seed"], window=w, patterns=pats), x.get("tail")) for w in self.windows}
        pols = F.tip_policies(max_vote=min(self.max_vote, nmin), with_guess=True, windows=self.windows)
        Js = JV.Judge(self.development_rows, tS, benches=self.benches, policies=pols, roles=SIGNAL_ROLES, settings=self.judge_settings)
        Jg = JV.Judge(self.development_rows, tJ, benches=self.benches, policies=pols, settings=self.judge_settings, hull_from={p[2] for p in Js.F.vertices},
                      hull_from_total={p[2] for p in Js.Ft.vertices})
        self._jg = dict(Js=Js, Jg=Jg, tS=tS, tJ=tJ, det=det, price=price, nmin=nmin)
        return self._jg

    def special_view(self, tip=None):
        t = tip or self.tip()
        return sorted(t["entries"] + t["special"], key=lambda e: e.get("pos", 10 ** 6))

    def judges_special(self):
        if not self.tip()["special"]:
            j = self.judges()
            return j["Js"], j["Jg"]
        if self._jsp is not None:
            return self._jsp
        V = self.special_view()
        self.load_routes(V, self.SIGR + self.JR)
        tS, tJ = self.arm(V, self.SIGR), self.arm(V, self.JR)
        nmin = min([len(tS.get(r, ())) for r in self.SIGR] + [len(tJ.get(r, ())) for r in self.JR])
        if nmin < self.min_tip_reads:
            return None
        pols = F.tip_policies(max_vote=min(self.max_vote, nmin), with_guess=False)
        Js = JV.Judge(self.development_rows, tS, benches=self.benches, policies=pols, roles=SIGNAL_ROLES, settings=self.judge_settings)
        Jg = JV.Judge(self.development_rows, tJ, benches=self.benches, policies=pols, settings=self.judge_settings, hull_from={p[2] for p in Js.F.vertices}, hull_from_total={p[2] for p in Js.Ft.vertices})
        self._jsp = (Js, Jg)
        return self._jsp

    def frontier_record(self):
        j = self.judges()
        Js, Jg = j["Js"], j["Jg"]
        V = [p[2] for p in sorted(Js.F.vertices)]
        pts = []
        for n in sorted(Jg.F.table, key=lambda n: F.macro_mean(Jg.F.table[n], self.development_rows, "tokens")):
            ts, tj = Js.F.table[n], Jg.F.table[n]
            pts.append(dict(name=n, vertex=n in V, multiple=F.macro_mean(tj, self.development_rows, "tokens") / Jg.F.base_tokens,
                            multiple_total=F.macro_mean(tj, self.development_rows, "total") / Jg.Ft.base_tokens,
                            acc_judge=F.macro_mean(tj, self.development_rows, "ok"), acc_signal=F.macro_mean(ts, self.development_rows, "ok")))
        default = self.tip()["policy"].get("default") or self.default_policy
        if default not in Jg.F.table:
            default = "vote1"
        return dict(vertices=V, points=pts, default=default, report=["vote1", default, V[-1]], base_tokens=Jg.F.base_tokens, base_total=Jg.Ft.base_tokens)

    def G(self):
        return self.state["gens"][str(self.g)]

    def step_signals(self):
        G, tip = self.G(), self.tip()["entries"]
        allr = self.SIGR + self.JR
        self.routes_for(tip, allr, "tip")
        miss = self.read(f"g{self.g:03d}_tip", self.reqs(tip, allr, range(self.tip_reads), prio=self.prio["tip_cache"]), "tip cache")
        if self.tip()["special"] and self.cfg["admission"]:
            V = self.special_view()
            self.routes_for(V, allr, "special")
            G["special_reads_missing"] = self.read(f"g{self.g:03d}_special", self.reqs(V, allr, range(self.tip_reads), prio=self.prio["specialisation_cache"]),
                                                   "specialisation system cache")
        j = self.judges()
        sig = {}
        for b in self.benches:
            rs = [r for r in self.SIGR if self.rows[r]["bench"] == b]
            xs = [x for r in rs for x in j["tS"][r]]
            dis = [len({F.cls_of(x) for x in j["tS"][r][:2]}) > 1 for r in rs if len(j["tS"][r]) >= 2]
            sig[b] = dict(rows=len(rs), acc=sum(x["ok"] for x in xs) / len(xs), tokens=sum(x["tokens"] for x in xs) / len(xs),
                          disagree=sum(dis) / max(len(dis), 1), guess={str(w): sum(bool(x["guess"][w]) for x in xs) / len(xs) for w in self.windows})
        det = j["det"]
        fr = self.frontier_record()
        self.tip()["policy"].update(frontier=fr, default=fr["default"], guess=dict(seed=det["seed"], patterns=det["patterns"], window=det["window"],
                                                                               windows=list(self.windows), lobo_value=det["lobo_value"], price=j["price"]))
        G["signals"] = dict(by_bench=sig, tip_reads_missing=miss, tip_reads_min=j["nmin"])
        self.log(f"  signals: tip {self.tip()['id']} reads per row >= {j['nmin']}; frontier vertices {fr['vertices']}; guess detector seed '{det['seed']}' + {det['patterns']}")

    def step_candidates(self):
        G, tip = self.G(), self.tip()
        cands, k = [], 0

        def cid():
            nonlocal k
            k += 1
            return f"g{self.g:03d}c{k - 1:02d}"

        def payload(entry):
            return json.dumps([PR.canon_ops(entry.get("ops")), entry.get("settings") or None], sort_keys=True)
        tried = {payload(c["entry"]) for gk, g in self.state["gens"].items() if int(gk) != self.g for c in g.get("cands", [])
                 if c.get("kind") in ("operator", "setting") and c.get("tip") == tip["id"] and (c.get("decided") or (c.get("screen") or {}).get("judged"))}
        for o in self.cfg.get("operators") or []:
            ops = PR.canon_ops(o["ops"])
            if payload(dict(ops=ops)) not in tried:
                cands.append(dict(cid=cid(), kind="operator", name=str(o.get("name") or "operator"), source=[], entry=dict(BASE_ENTRY, ops=ops, author="operator")))
        cur = self.settings_of([e for e in tip["entries"] if e["cond"] == "always"])
        for o in self.cfg.get("settings_arm") or []:
            st_ = dict(o["settings"])
            if "effort" in " ".join(st_).lower() or "reasoning" in " ".join(st_).lower():
                raise Refuse(f"settings_arm {o}: the reasoning effort is never a knob")
            if all(cur.get(k_) == v for k_, v in st_.items()) or payload(dict(settings=st_)) in tried:
                continue
            cands.append(dict(cid=cid(), kind="setting", name=str(o.get("name") or json.dumps(st_, sort_keys=True)), source=[], entry=dict(BASE_ENTRY, settings=st_, author="settings_arm"),
                              **({"view_only": True} if o.get("view_only") else {})))
        if int(self.prop["cards_per_generation"]) > 0 and not self.offline:
            cands += self.propose(cid, cands)
        if self.cfg["audit"]:
            cond = [e for e in tip["entries"] if e["cond"] != "always" and e.get("audit") != "GENERAL"]
            groups = _read_json(self.cfg["audit_groups"]) if self.cfg.get("audit_groups") else {}
            used = {c_ for v in groups.values() for c_ in v}
            units = [(n, [e for e in cond if e["cid"] in set(v)]) for n, v in groups.items()] + [(e["cid"], [e]) for e in cond if e["cid"] not in used]
            for n, es in units:
                if es:
                    cands.append(dict(cid=cid(), kind="audit", name=f"audit {n}", source=sorted({b for e in es for b in e.get("source") or []}), members=[e["cid"] for e in es]))
        for e in tip["special"]:
            seen_b = (e.get("verdict") or {}).get("benches")
            if self.retest == "all" or (self.retest == "not_covered_by_verdict" and seen_b and set(self.benches) - set(seen_b)):
                cands.append(dict(cid=cid(), kind="retest", name=f"retest {e['cid']}", source=list(e.get("source") or []),
                                  entry={k_: v for k_, v in e.items() if k_ not in ("verdict",)}, member=e["cid"]))
        for c in cands:
            c.update(tip=tip["id"], gen=self.g, status="candidate")
        G["cands"] = cands
        self.log("  candidates: " + ", ".join(f"{n} {v}" for n, v in sorted(collections.Counter(c["kind"] for c in cands).items())))

    def proposer_requests(self):
        tip = self.tip()["entries"]
        tv = self.tip_view()
        parent_show = {r: self.samples(tip, r) for role in PROPOSER_ROLES for r in self.role[role]}
        n_cards = int(self.prop["cards_per_generation"])
        n_wrong, n_right = int(self.prop["wrong_per_card"]), int(self.prop["right_per_card"])
        n_mixed = n_cards // 2
        alloc = self.alloc_sources(n_cards - n_mixed, self.residual(parent_show))
        seed_single = self.prop["card_seed_single"].format(generation=self.g)
        seed_mixed = self.prop["card_seed_mixed"].format(generation=self.g)
        cards = [(b, PR.cards_source(self.rows, parent_show, b, i, seed=seed_single, n_wrong=n_wrong, n_right=n_right))
                 for b in sorted(alloc) for i in range(alloc[b])]
        cards += [(None, PR.cards_mixed(self.rows, parent_show, k_, seed=seed_mixed, n_wrong=n_wrong, n_right=n_right)) for k_ in range(n_mixed)]
        reqs, props = [], []
        for b, ids in cards:
            if not ids:
                continue
            system, user = PR.card_prompt(self.rows, parent_show, ids, tv, self.menu)
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user + "\n\n" + PR.RULE_NOTE}]
            key = f"chat|{self.run_name}|g{self.g:03d}|card{len(reqs)}|{sha16(json.dumps(msgs))[:8]}"
            reqs.append(dict(key=key, mode="chat", priority=self.prio["chat"], messages=msgs, temperature=float(self.prop["temperature"]),
                             max_tokens=int(self.prop["max_tokens"])))
            props.append(dict(key=key, card=ids, source=[b] if b else [], mixed=b is None))
        return reqs, props, alloc

    def propose(self, cid, so_far):
        G = self.G()
        tv = self.tip_view()
        reqs, props, alloc = self.proposer_requests()
        self.io.submit(f"g{self.g:03d}_chat", reqs)
        self.wait([q["key"] for q in reqs], "proposer")
        seen = {PR.entry_key(e) for e in tv["entries"]} | {PR.entry_key(e) for e in self.tip()["special"]}
        have_set = {json.dumps(c["entry"]["settings"], sort_keys=True) for c in so_far if c["kind"] == "setting"}
        have_set |= {json.dumps(c["entry"]["settings"], sort_keys=True) for gk, g_ in self.state["gens"].items() if int(gk) != self.g for c in g_.get("cands", [])
                     if c.get("kind") == "setting" and c.get("tip") == self.tip()["id"] and (c.get("screen") or {}).get("judged")}
        out, dropped = [], []
        for p in props:
            x = self.io.get(p["key"])
            cand, err = PR.check_answer(x, tv, self.menu, self.rows, p["card"])
            proc = (cand or {}).get("procedure") if isinstance((cand or {}).get("procedure"), str) and (cand or {}).get("procedure").strip() else None
            sett = (cand or {}).get("settings") or {}
            ops = PR.canon_ops((cand or {}).get("ops"))
            kind = None
            if cand is not None and err is None:
                always = PR.is_always(cand["condition"])
                if sett and not proc and not ops and always:
                    kind = "setting"
                elif proc or ops or sett:
                    kind = "entry"
                else:
                    err = "empty proposal"
            if kind == "entry" and PR.entry_key(cand) in seen:
                kind, err = None, "duplicate of a tip or specialisation entry or of an earlier candidate"
            if kind == "setting" and json.dumps(sett, sort_keys=True) in have_set:
                kind, err = None, "duplicate of a settings-arm cell (this generation, or read on this tip before)"
            if kind is None:
                dropped.append(dict(why=str(err)[:160], source=p["source"], mixed=p["mixed"], head=((x or {}).get("content") or "")[-200:] if cand is None else
                                    json.dumps({k_: cand.get(k_) for k_ in ("condition", "procedure", "settings", "ops") if cand.get(k_)}, ensure_ascii=False)[:200]))
                continue
            c = cid()
            if kind == "entry":
                seen.add(PR.entry_key(cand))
                always = PR.is_always(cand["condition"])
                what = "+".join(w for w, v_ in (("text", proc), ("settings", sett), ("ops", ops)) if v_)
                out.append(dict(cid=c, kind="entry", name=f"entry {c} ({'always-on' if always else 'routed'} {what})", source=list(p["source"]), card=p["card"], mixed=p["mixed"],
                                why=cand.get("why", ""), entry=dict(condition="always" if always else cand["condition"], cond="always" if always else sha16(cand["condition"]),
                                                                    procedure=proc, settings=dict(sett), ops=ops, cid=c, author="proposer", source=list(p["source"]))))
            else:
                have_set.add(json.dumps(sett, sort_keys=True))
                out.append(dict(cid=c, kind="setting", name="proposer " + json.dumps(sett, sort_keys=True), source=[], proposed_from=list(p["source"]), card=p["card"],
                                why=cand.get("why", ""), entry=dict(BASE_ENTRY, settings=dict(sett), author="proposer")))
        G["dropped"] = dropped
        G["cards"] = dict(single=len([p for p in props if not p["mixed"]]), mixed=len([p for p in props if p["mixed"]]), alloc=alloc)
        self.log(f"  proposer: {len(props)} cards ({G['cards']['single']} single-benchmark, {G['cards']['mixed']} mixed) -> {sum(c['kind'] == 'entry' for c in out)} entries "
                 f"({sum(c['kind'] == 'entry' and c['entry']['cond'] == 'always' for c in out)} always-on), {sum(c['kind'] == 'setting' for c in out)} settings; "
                 f"dropped {len(dropped)}: {dict(collections.Counter(d['why'][:50] for d in dropped))}")
        return out

    def cand_view(self, c):
        tip = self.tip()["entries"]
        if c["kind"] == "audit":
            return [e for e in tip if e["cid"] not in set(c["members"])]
        return tip + [c["entry"]]

    def cand_rows(self, c, rows, screen):
        if c["kind"] in ("operator", "setting"):
            return list(rows)
        if c["kind"] == "audit":
            mem = [e for e in self.tip()["entries"] if e["cid"] in set(c["members"])]
            return [r for r in rows if any(self.fires(e, r) for e in mem)]
        rr = self.fired_rows(c["entry"], rows)
        card = set(c.get("card") or ()) if screen else set()
        return [r for r in rr if not (screen and self.rows[r]["bench"] in set(c.get("source") or [])) and r not in card]

    def to_judge(self, Jx, c, rows, upto):
        on_judge_rows = Jx.roles == (VALIDATION_ROLE,)
        rr = self.cand_rows(c, rows, screen=not on_judge_rows)
        salt = self.key_salt(c)
        arm = self.arm(self.cand_view(c), rr, upto, salt=salt)
        miss = [r for r in rr if r not in arm]
        if miss:
            why = {r: self.unreadable(self.cand_view(c), r, upto, salt) for r in miss}
            if any(v is None for v in why.values()):
                return None
            c.setdefault("skipped_rows", {})["judge" if on_judge_rows else "signal"] = dict(
                n=len(miss), why=dict(collections.Counter(why.values()).most_common(3)), by_bench=dict(collections.Counter(self.rows[r]["bench"] for r in miss)))
            rr = [r for r in rr if r in arm]
        if c["kind"] == "audit":
            tipa = {r: Jx.tip[r] for r in rr}
            return Jx.cand_from_arm(c["name"], tipa, "entry", source=c["source"], routed=rr, base=Jx.system_of_arm(arm, routed=rr))
        kind = "entry" if c["kind"] == "retest" else c["kind"]
        return Jx.cand_from_arm(c["name"], arm, kind, source=c["source"], routed=None if (c["kind"] == "operator" and not miss) else rr)

    def unreadable(self, entries, r, upto, salt=None):
        _, _, sg = self.view(entries, r)
        for s_ in range(upto):
            k = self.read_key(sg, r, s_, salt)
            x = self.io.get(k)
            if x is not None and x.get("op_skipped"):
                return "op_skipped: " + str(x.get("op_skipped"))[:60]
            if x is None and k in self.io.err and not self.io.err[k].get("image_limit"):
                return "error: " + str(self.io.err[k].get("error"))[:60]
        return None

    def budgets(self, Jx, name, arm):
        out = {}
        n = min(len(arm[r]) for r in Jx.J)
        for lab, pol in F.budget_policies(name):
            if len(pol["init"]) + len(pol["extra_views"]) > n:
                continue
            out[lab] = self.brief(Jx.score(Jx.cand_from_views(f"{name} {lab}", pol, {name: arm}, kind="setting")))
        return out

    @staticmethod
    def brief(r):
        if not r.get("judged"):
            return dict(judged=False, error=r.get("error"))
        e = r["eq"]
        out = dict(judged=True, acc=r["acc"], multiple=r["multiple"], multiple_total=(r.get("total") or {}).get("multiple"), raw=r["raw"]["d"], d=e["d"], se=e["se"],
                   p_macro=e["p_macro"], p_gen=e["p_gen"], carrier=e["carrier"], npos=e["npos"], k=e["k"], scored=r["scored"], ref=r.get("ref"),
                   per_bench={b: round(v["d"], 5) for b, v in e["per_bench"].items()}, d_total=(r.get("total") or {}).get("d"), pooled=r["pooled"]["d"],
                   pooled_p=r["pooled"]["p_macro"])
        for k_ in ("accept", "reason", "layer", "K", "holm_threshold", "gate"):
            if k_ in r:
                out[k_] = r[k_]
        return out

    def _passes(self, r):
        return bool(r.get("judged") and r["eq"]["d"] > 0 and r["eq"]["p_macro"] <= self.screen_p)

    def step_prescreen(self):
        G, j = self.G(), self.judges()
        Js = j["Js"]
        comp = [c for c in G["cands"] if c["kind"] in ("operator", "entry", "retest")]
        sets = [c for c in G["cands"] if c["kind"] == "setting"]
        text = [c for c in comp if c["kind"] in ("entry", "retest")]
        if text:
            self.routes_for([c["entry"] for c in text], self.SIGR + self.JR, "cands")
        reqs = []
        for c in comp:
            reqs += self.reqs(self.cand_view(c), self.cand_rows(c, self.SIGR, True), range(self.s_screen))
        for c in sets:
            reqs += self.reqs(self.cand_view(c), self.SIGR, range(self.s_screen))
        self.read(f"g{self.g:03d}_screen", reqs, "pre-screen")
        passed = []
        more = []
        for c in sets:
            arm = self.arm(self.cand_view(c), self.SIGR)
            n_ = min([len(arm.get(r, ())) for r in self.SIGR])
            if 1 <= n_ < self.arm_min_reads and not c.get("view_only"):
                x1 = self.brief(Js.score(Js.cand_from_views(f"{c['name']} x1.00", F.views_policy([c["name"]]), {c["name"]: arm}, kind="setting")))
                if not (x1.get("judged") and x1["d"] > 0):
                    c.update(status=f"screened out at the first rung ({n_} reads per signal row; one read of the arm against the tip)", screen=dict(x1, passed=False, rung=1))
                    continue
            if n_ < self.tip_reads:
                more += self.reqs(self.cand_view(c), self.SIGR, range(self.tip_reads))
        if more:
            self.read(f"g{self.g:03d}_screen2", more, "pre-screen, full cache of the settings arms that passed the first rung")
        views = {}
        for c in sets:
            if (c.get("screen") or {}).get("rung") == 1:
                continue
            arm = self.arm(self.cand_view(c), self.SIGR)
            if any(len(arm.get(r, ())) < self.arm_min_reads for r in self.SIGR):
                c.update(status="pre-screen reads missing", screen=dict(judged=False))
                continue
            views[c["name"]] = arm
            c["budgets_signal"] = self.budgets(Js, c["name"], arm)
            sc = dict(c["budgets_signal"]["default"])
            ok = bool(sc["judged"] and sc["d"] > 0 and sc["p_macro"] <= self.screen_p) and not c.get("view_only")
            c["screen"] = dict(sc, passed=ok)
            c["status"] = ("a cached view for view-diverse votes only (not a settings candidate)" if c.get("view_only") else
                           "passed the pre-screen (default budget)" if ok else "screened out on signal rows (default budget)")
            if ok:
                passed.append((sc["p_macro"], c))
        vn = ["tip"] + sorted(views)
        for i_, va in enumerate(vn):
            for vb in vn[i_ + 1:]:
                best = None
                for pol in F.grammar_views([va, vb]):
                    r = Js.score(Js.cand_from_views(F.views_name(pol), pol, views))
                    if self._passes(r) and (best is None or r["eq"]["p_macro"] < best[0]["eq"]["p_macro"]):
                        best = (r, pol)
                if best is not None:
                    pc = dict(cid=f"g{self.g:03d}view{len([c for c in G['cands'] if c['kind'] == 'viewpolicy']):02d}", kind="viewpolicy", name=F.views_name(best[1]),
                              source=[], policy=best[1], views=sorted({v for v in best[1]["init"] + best[1]["extra_views"] if v != "tip"}), tip=self.tip()["id"], gen=self.g,
                              status="passed the pre-screen", screen=dict(self.brief(best[0]), passed=True))
                    G["cands"].append(pc)
                    passed.append((best[0]["eq"]["p_macro"], pc))
        for c in comp:
            if c["kind"] in ("entry", "retest") and not self.cand_rows(c, self.SIGR, True):
                c.update(status="no off-source rows routed (its condition matches only its source benchmark)", screen=dict(judged=False, no_off_source=True, passed=False))
                continue
            jc = self.to_judge(Js, c, self.SIGR, self.tip_reads)
            if jc is None:
                c.update(status="pre-screen reads missing", screen=dict(judged=False))
                continue
            r = Js.score(jc)
            c["screen"] = self.brief(r)
            ok = self._passes(r)
            c["screen"]["passed"] = ok
            c["status"] = "passed the pre-screen" if ok else "screened out on signal rows"
            if ok:
                passed.append((r["eq"]["p_macro"], c))
        for c in [c for c in comp if c["kind"] == "operator" and c.get("screen", {}).get("judged")]:
            ex = self.arm(self.cand_view(c), self.SIGR)
            best = None
            for pol in F.grammar((c["cid"],), windows=self.windows):
                if pol["trigger"] == "none":
                    continue
                r = Js.score(Js.cand_from_policy(pol, extra_arm=ex))
                if self._passes(r) and (best is None or r["eq"]["p_macro"] < best[0]["eq"]["p_macro"]):
                    best = (r, pol)
            if best is not None:
                pc = dict(cid=f"{c['cid']}p", kind="policy", name=F.name_of(best[1]), source=[], policy=best[1], extra=c["cid"], tip=c["tip"], gen=self.g,
                          status="passed the pre-screen", screen=dict(self.brief(best[0]), passed=True))
                G["cands"].append(pc)
                passed.append((best[0]["eq"]["p_macro"], pc))
        passed.sort(key=lambda x: x[0])
        for i, (_, c) in enumerate(passed):
            c["sent"] = i < self.k_max
            if not c["sent"]:
                c["status"] = f"passed the pre-screen, not sent (more than {self.k_max} candidates)"
        self.log(f"  pre-screen: {len(comp)} components read on signal rows; passed {[c['name'] for _, c in passed]}; sent {sum(bool(c.get('sent')) for _, c in passed)}")

    def step_judge(self):
        G, j = self.G(), self.judges()
        Jg = j["Jg"]
        by = {c["cid"]: c for c in G["cands"]}
        sent = [c for c in G["cands"] if c.get("sent")]
        audits = [c for c in G["cands"] if c["kind"] == "audit"]
        reqs = []
        byname = {c["name"]: c for c in G["cands"] if c["kind"] == "setting"}
        for c in sent:
            if c["kind"] in ("setting", "viewpolicy"):
                for x in ([c] if c["kind"] == "setting" else [byname[v] for v in c["views"]]):
                    reqs += self.reqs(self.cand_view(x), self.JR, range(self.tip_reads))
                continue
            x = by[c["extra"]] if c["kind"] == "policy" else c
            reqs += self.reqs(self.cand_view(x), self.cand_rows(x, self.JR, False), range(self.s_judge))
        for c in audits:
            reqs += self.reqs(self.cand_view(c), self.cand_rows(c, self.JR, False), range(self.s_audit), salt=self.key_salt(c))
        self.read(f"g{self.g:03d}_judge", reqs, "JUDGE reads")
        jcs, cs = [], []
        for c in sent:
            if c["kind"] in ("setting", "viewpolicy"):
                va = {x["name"]: self.arm(self.cand_view(x), self.JR) for x in ([c] if c["kind"] == "setting" else [byname[v] for v in c["views"]])}
                if any(len(a_.get(r, ())) < self.arm_min_reads for a_ in va.values() for r in self.JR):
                    jc = None
                elif c["kind"] == "setting":
                    c["budgets_judge"] = self.budgets(Jg, c["name"], va[c["name"]])
                    jc = Jg.cand_from_views(c["name"], F.budget_policies(c["name"])[1][1], va, kind="setting")
                else:
                    jc = Jg.cand_from_views(c["name"], c["policy"], va)
            elif c["kind"] == "policy":
                ex = self.arm(self.cand_view(by[c["extra"]]), self.JR)
                jc = Jg.cand_from_policy(c["policy"], extra_arm=ex, name=c["name"]) if all(r in ex for r in self.JR) else None
            else:
                jc = self.to_judge(Jg, c, self.JR, self.tip_reads)
            if jc is None:
                c.update(status="JUDGE reads missing", decided=False)
                continue
            jcs.append(jc)
            cs.append(c)
        recs = Jg.decide(jcs, family=f"generation {self.g}") if jcs else []
        for c, r in zip(cs, recs):
            c.update(judge=self.brief(r), accept=bool(r["accept"]), decided=True, status=("ACCEPTED: " if r["accept"] else "rejected: ") + str(r["reason"]),
                     line=JV.line(r, self.benches))
        ja, ca = [], []
        for c in audits:
            jc = self.to_judge(Jg, c, self.JR, self.tip_reads)
            if jc is None:
                c.update(status="audit reads missing", decided=False)
                continue
            ja.append(jc)
            ca.append(c)
        for c, r in zip(ca, Jg.audit_entries(ja) if ja else []):
            c.update(judge=self.brief(r), layer=r["layer"], accept=r["layer"] == "GENERAL", decided=True, status=f"{r['layer']}: {r['reason']}", line=JV.line(r, self.benches))
        for c in cs:
            s_, j_ = c.get("screen") or {}, c.get("judge") or {}
            if s_.get("judged") and j_.get("judged"):
                self.shrinkage(dict(gen=self.g, tip=self.tip()["id"], cid=c["cid"], kind=c["kind"], name=c["name"],
                                    signal_d=s_["d"], signal_se=s_["se"], judge_d=j_["d"], judge_se=j_["se"], signal_raw=s_["raw"], judge_raw=j_["raw"],
                                    kept=(j_["d"] / s_["d"]) if s_["d"] > 0 else None, multiple=j_["multiple"], accepted=bool(c.get("accept")), t=time.time()))
        self.admission()
        self.log(f"  judge: family K {len(recs)}: accepted {[c['name'] for c in cs if c.get('accept')]}; audit {[(c['name'], c.get('layer')) for c in ca]}"
                 f"; admitted to the specialisation layer {[c['cid'] for c in G['cands'] if (c.get('admit') or {}).get('admitted')]}")

    def shrinkage(self, row):
        self.state["shrinkage"].append(row)
        p = os.path.join(self.rd, "shrinkage.jsonl")
        if row_key(row) in report.written_keys(p, row_key):
            return
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def reload_routes(self):
        t = self.tip()
        G = self.state["gens"].get(str(self.g)) or {}
        ents = list(t["entries"]) + list(t["special"]) + [c["entry"] for c in G.get("cands", []) if isinstance(c.get("entry"), dict)]
        self.load_routes(ents, self.SIGR + self.JR)

    def admission(self):
        G = self.G()
        adm = [c for c in G["cands"] if c["kind"] == "entry" and c.get("source") and not c.get("accept")
               and (c.get("decided") or (c.get("screen") or {}).get("no_off_source") or ((c.get("screen") or {}).get("judged") and not c["screen"].get("passed")))]
        if not self.cfg["admission"] or not adm:
            return
        jj = self.judges_special()
        if jj is None:
            for c in adm:
                c["admit"] = dict(judged=False, admitted=False, why="the specialisation system's cache is not filled")
            self.log("  admission: the specialisation system's cache is not filled; no entry is tested this generation")
            return
        Jss, Jsg = jj
        V = self.special_view()
        mon = {i for role in MONITOR_ROLES for i in self.role[role]}
        reqs, rowsets = [], {}
        for c in adm:
            src = set(c["source"])
            fj = [r for r in self.fired_rows(c["entry"], self.JR) if self.rows[r]["bench"] in src]
            fm = [r for r in self.fired_rows(c["entry"], self.SIGR) if self.rows[r]["bench"] in src and r in mon]
            rowsets[c["cid"]] = (fj, fm)
            reqs += self.reqs(V + [c["entry"]], fj + fm, range(self.s_judge))
        self.read(f"g{self.g:03d}_admit", reqs, "admission reads (source benchmark)")
        tested = []
        for c in adm:
            fj, fm = rowsets[c["cid"]]
            aj, am = self.arm(V + [c["entry"]], fj), self.arm(V + [c["entry"]], fm)
            if not fj or any(r not in aj for r in fj):
                c["admit"] = dict(judged=False, admitted=False, why="no routed JUDGE row in the source benchmark" if not fj else "admission reads missing")
                continue
            e = Jsg.source_effect(Jsg.cand_from_arm(c["name"], aj, "entry", routed=fj), c["source"])
            m = (Jss.source_effect(Jss.cand_from_arm(c["name"], am, "entry", routed=fm), c["source"], rows=mon)
                 if fm and all(r in am for r in fm) else dict(judged=False, error="no routed MONITOR row in the source benchmark"))
            c["admit"] = dict(judged=bool(e.get("judged")), admitted=False, source=c["source"], judge=e, monitor=m, routed_judge=len(fj), routed_monitor=len(fm))
            if e.get("judged"):
                tested.append(c)
        acc = ST.holm([c["admit"]["judge"]["p"] for c in tested], Jsg.alpha) if tested else []
        order = sorted(range(len(tested)), key=lambda i: tested[i]["admit"]["judge"]["p"])
        for rank, i in enumerate(order):
            tested[i]["admit"]["holm_threshold"] = Jsg.alpha / (len(tested) - rank)
        for c, a_ in zip(tested, acc):
            e, m = c["admit"]["judge"], c["admit"]["monitor"]
            ok = bool(a_ and e["d"] > 0)
            c["admit"].update(admitted=ok, K=len(tested))
            c["status"] += (f" | admission on the source benchmark {c['source']}: {100 * e['d']:+.2f} +- {100 * e['se']:.2f} pp on its JUDGE rows (p {e['p']:.4f}, Holm {c['admit']['holm_threshold']:.4f}, K {len(tested)})"
                            + (f"; MONITOR rows {100 * m['d']:+.2f} +- {100 * m['se']:.2f}" if m.get("judged") else "; MONITOR rows: none routed")
                            + (" -> ID_SPECIFIC, specialisation layer (never a general gain)" if ok else " -> not admitted"))
            s_ = c.get("screen") or {}
            self.shrinkage(dict(gen=self.g, tip=self.tip()["id"], cid=c["cid"], kind="admission", name=c["name"], source=c["source"],
                                signal_d=s_.get("d"), signal_se=s_.get("se"), judge_d=(c.get("judge") or {}).get("d"), judge_se=(c.get("judge") or {}).get("se"),
                                source_judge_d=e["d"], source_judge_se=e["se"], source_judge_p=e["p"], source_monitor_d=m.get("d"), source_monitor_se=m.get("se"),
                                kept=(m["d"] / e["d"]) if (m.get("judged") and e["d"] > 0) else None, admitted=ok, accepted=False, t=time.time()))
        G["admission"] = dict(reference="the specialisation system (general + layer)" if self.tip()["special"] else "the general system (the layer is empty)",
                              layer_entries=len(self.tip()["special"]), tested=len(tested), admitted=[c["cid"] for c in tested if c["admit"]["admitted"]],
                              reference_one_read=sum(Jsg.tip_v.values()) / len(Jsg.tip_v))
        self.log(f"  admission: {len(tested)} failed entries tested on their source benchmark against {G['admission']['reference']}; admitted {G['admission']['admitted']}")

    def step_update(self):
        G, tip = self.G(), self.tip()
        ents, special, pol = list(tip["entries"]), list(tip["special"]), json.loads(json.dumps(tip["policy"]))
        by = {c["cid"]: c for c in G["cands"]}
        change = []
        for c in G["cands"]:
            if (c.get("admit") or {}).get("admitted"):
                e, m = c["admit"]["judge"], c["admit"]["monitor"]
                off = c.get("judge") if (c.get("judge") or {}).get("judged") else (c.get("screen") or {})
                special.append(dict(c["entry"], cid=c["cid"], source=c["source"], pos=10 ** 5 + self.g * 100 + len(special),
                                    verdict=dict(layer="ID_SPECIFIC", gen=self.g, cid=c["cid"], benches=list(self.benches), basis="admission test on the JUDGE rows of the source benchmark",
                                                 d=off.get("d"), p_gen=off.get("p_gen"), off_source_rows="JUDGE" if off is c.get("judge") else "signal (screened out)",
                                                 source_d=e["d"], source_se=e["se"], source_p=e["p"], K=c["admit"].get("K"),
                                                 monitor_d=m.get("d"), monitor_se=m.get("se"))))
                change.append(f"+specialisation {c['cid']} (ID_SPECIFIC on {','.join(c['source'])})")
                continue
            if not c.get("decided"):
                continue
            v = dict(gen=self.g, cid=c["cid"], benches=list(self.benches), **{k: c["judge"].get(k) for k in ("d", "se", "p_gen", "pooled", "pooled_p", "scored", "multiple")})
            if c["kind"] == "viewpolicy" and c["accept"]:
                bn = {x["name"]: x["entry"] for x in G["cands"] if x["kind"] == "setting"}
                pol["extras"].append(dict(cid=c["cid"], policy=c["policy"], views={v_: bn[v_] for v_ in c["views"]}, name=c["name"], verdict=dict(v, layer="GENERAL")))
                change.append(f"+view policy {c['name']}")
            elif c["kind"] in ("operator", "entry", "setting") and c["accept"]:
                ents.append(dict(c["entry"], cid=c["cid"], source=c["source"], accepted=self.g, audit="GENERAL", verdict=dict(v, layer="GENERAL")))
                change.append(f"+{c['kind']} {c['name']}")
            elif c["kind"] == "policy" and c["accept"]:
                pol["extras"].append(dict(cid=c["cid"], policy=c["policy"], extra=by[c["extra"]]["entry"], name=c["name"], verdict=dict(v, layer="GENERAL")))
                change.append(f"+policy {c['name']}")
            elif c["kind"] == "retest" and c["accept"]:
                e = next(e for e in special if e["cid"] == c["member"])
                special.remove(e)
                ents.append(dict({k: x for k, x in e.items() if k != "verdict"}, audit="GENERAL", verdict=dict(v, layer="GENERAL")))
                change.append(f"promoted {c['member']}")
            elif c["kind"] == "retest":
                next(e for e in special if e["cid"] == c["member"]).setdefault("retests", []).append(dict(v, accept=False))
            elif c["kind"] == "audit":
                for e in [e for e in ents if e["cid"] in set(c["members"])]:
                    if c["layer"] == "GENERAL":
                        e.update(audit="GENERAL", verdict=dict(v, layer="GENERAL"))
                    else:
                        ents.remove(e)
                        special.append(dict({k: x for k, x in e.items() if k != "audit"}, verdict=dict(v, layer=c["layer"], audit=c["name"])))
                        change.append(f"{e['cid']} -> specialisation ({c['layer']})")
        G["tip_change"] = change
        if not change:
            self.log(f"  tip {tip['id']} unchanged")
            return
        self.state["tips"].append(dict(id=tip["id"], born=tip.get("born"), entries=len(tip["entries"]), special=len(tip["special"]), replaced=self.g))
        special.sort(key=lambda e: e.get("pos", 10 ** 6))
        new = dict(id=tip_id(self.g), born=self.g, parent=tip["id"], entries=ents, special=special, policy=dict(pol, frontier=None))
        self.state["tip"] = new
        self._jg, self._jsp = None, None
        self.write_tip(new)
        allr = self.SIGR + self.JR
        if not self.offline:
            self.ensure_routes(ents, allr, "newtip")
            n = self.io.submit(f"g{self.g:03d}_newtip", self.reqs(ents, allr, range(self.tip_reads), prio=self.prio["new_tip_cache"]))
            if special and self.cfg["admission"]:
                V = self.special_view(new)
                self.ensure_routes(V, allr, "newspecial")
                G["new_special_cache_requested"] = self.io.submit(f"g{self.g:03d}_newspecial", self.reqs(V, allr, range(self.tip_reads), prio=self.prio["new_specialisation_cache"]))
        else:
            self.load_routes(ents, allr)
            n = None
        have = collections.Counter(len(v) for v in self.arm(ents, allr).values())
        G["new_tip_cache"] = dict(requested=n, rows_by_reads={str(k_): v for k_, v in sorted(have.items())}, rows_without=len(allr) - sum(have.values()),
                                  missing_reads=sum(self.tip_reads - k_ for k_, v in have.items() for _ in range(v)) + self.tip_reads * (len(allr) - sum(have.values())))
        self.log(f"  tip {tip['id']} -> {new['id']}: {'; '.join(change)}; cache of the new tip: {G['new_tip_cache']}")

    def write_tip(self, tip):
        os.makedirs(os.path.join(self.rd, "tips"), exist_ok=True)
        p = os.path.join(self.rd, "tips", tip["id"] + ".json")
        jdump(p, tip)
        return p

    def step_report(self):
        report.write(self)

    def generation(self, g, stop_before=None):
        self.g = g
        G = self.state["gens"].setdefault(str(g), dict(t0=time.time(), done=[], times={}, tip_at_start=self.tip()["id"]))
        self.save()
        if G["done"]:
            self.reload_routes()
        for st in STEPS:
            if st in G["done"]:
                continue
            if stop_before and st == stop_before:
                self.log(f"stop before step {st}")
                return False
            t = time.time()
            self.log(f"generation {g} step {st}")
            getattr(self, "step_" + st)()
            if st == "signals":
                G["frontier"] = self.tip()["policy"]["frontier"]
            G["done"].append(st)
            G["times"][st] = round(time.time() - t, 1)
            self.save()
        self.state["gen"] = g
        G["t_end"] = time.time()
        self.save()
        return True

    def run(self, generations=1, stop_before=None):
        ran = []
        g = max([int(k) for k in self.state["gens"]] or [0])
        n = int(generations)
        if g and len(self.state["gens"][str(g)]["done"]) < len(STEPS):
            if not self.generation(g, stop_before):
                return ran
            ran.append(g)
            n -= 1
        for _ in range(max(0, n)):
            if os.path.exists(os.path.join(self.rd, "STOP")):
                self.log("STOP file: stopping at the generation boundary")
                return ran
            g = max([int(k) for k in self.state["gens"]] or [0]) + 1
            if not self.generation(g, stop_before):
                return ran
            ran.append(g)
        return ran
