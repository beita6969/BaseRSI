import glob
import gzip
import hashlib
import io
import json
import os

from basersi import config
from basersi import fingerprint as FP
from basersi.data.rows import ROLE_FIELD
from basersi.engine import grader
from basersi.engine import queue_file as QF
from basersi.engine.dispatch import aborted


def _key_records(raw, keys):
    if raw.find(b'{"key": "', 1) > 0 or not raw.endswith(b"\n"):
        x, rescued = QF.parse_line(raw.rstrip(b"\n"))
        return [r for r in ([x] if x is not None else rescued) if r.get("key") in keys]
    if not raw.startswith(b'{"key": "'):
        return []
    end = raw.find(b'"', 9)
    if end < 0 or raw[9:end].decode("utf-8", "replace") not in keys:
        return []
    x, rescued = QF.parse_line(raw.rstrip(b"\n"))
    return [x] if x is not None else rescued


def scan(queue_dir, keys, status=None):
    got = {}
    for f in sorted(glob.glob(os.path.join(queue_dir, "q", "out", "*.jsonl"))):
        with open(f, "rb") as fh:
            for raw in fh:
                for x in _key_records(raw, keys):
                    k = x["key"]
                    ok = not x.get("error") and x.get("content") is not None and not aborted(x)
                    if ok and k not in got:
                        got[k] = x
                    if status is not None:
                        why = "read" if ok else ("skipped", str(x["op_skipped"]).split(":")[0]) if x.get("op_skipped") else "error" if x.get("error") else "missing"
                        if _rank(why) > _rank(status.get(k, "missing")):
                            status[k] = why
    return got


def _rank(st):
    return dict(missing=0, error=1, read=3).get(st, 2) if isinstance(st, str) else 2


def count_reads(keys, status):
    out = dict(expected=len(keys), missing_reads=0, errored_reads=0, skipped_reads={})
    for k in keys:
        st = status.get(k, "missing")
        if st == "missing":
            out["missing_reads"] += 1
        elif st == "error":
            out["errored_reads"] += 1
        elif isinstance(st, tuple):
            out["skipped_reads"][st[1]] = out["skipped_reads"].get(st[1], 0) + 1
    return out


def class_text(c):
    if isinstance(c, tuple):
        inner = [class_text(x) for x in c]
        return "(" + ", ".join(inner) + ("," if len(inner) == 1 else "") + ")"
    if isinstance(c, frozenset):
        return "frozenset({" + ", ".join(sorted(class_text(x) for x in c)) + "})" if c else "frozenset()"
    if isinstance(c, list):
        return "[" + ", ".join(class_text(x) for x in c) + "]"
    return repr(c)


def read_line(rows, arm, x, row, s, **kw):
    r = rows[row]
    p = grader.record_pred(x, r)
    d = dict(arm=arm, row=row, s=int(s), ok=bool(x.get("ok")), pred=p, cls=class_text(grader.answer_class(p, r)), tokens=x.get("tokens"),
             prompt_tokens=x.get("prompt_tokens"), finish=x.get("finish"))
    d.update(kw)
    return d


def write(path, lines):
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, io.TextIOWrapper(gz, encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return dict(path=path, lines=len(lines), sha256=hashlib.sha256(open(path, "rb").read()).hexdigest())


class JudgeExport:
    def __init__(self, run, out_dir, export_config=None, judge_spec=None):
        self.run = run
        self.out = out_dir
        self.cfg = export_config or run.cfg["judge_export"]
        self.spec = judge_spec or config.load("judge")["exports"]
        os.makedirs(out_dir, exist_ok=True)

    def system(self):
        t = self.run.tip()
        if self.cfg["system"] == "general":
            return list(t["entries"])
        return self.run.special_view()

    def base_entries(self):
        return [e for e in self.run.tip()["entries"] if e["cond"] == "always"]

    def base_arm(self):
        ents = self.base_entries()
        st = self.run.settings_of(ents)
        return dict(arm=self.spec["floor_and_stock"]["floor_arm"], settings={k: v for k, v in sorted(st.items()) if v != self.run.stock.get(k)},
                    entries=[e.get("cid") for e in ents], text_entries=sum(1 for e in ents if e.get("procedure") or e.get("ops")),
                    origin="the tip's general part: its always-on general entries, without routed entries and without the specialisation layer")

    def keys_of(self, entries, rows, samples, salt=None):
        out = {}
        for r in rows:
            _, _, sg = self.run.view(entries, r)
            for s in samples:
                out[self.run.read_key(sg, r, s, salt)] = (r, s)
        return out

    def request(self, name, entries, rows, samples, salt=None):
        if not rows:
            return 0
        return self.run.read(f"export_{name}", self.run.reqs(entries, rows, samples, prio=self.run.prio["candidate"], salt=salt), f"export {name}")

    def removal_salt(self, unit):
        fmt = self.cfg.get("removal_key_salt")
        return fmt.format(unit=unit) if fmt else None

    def units(self, ents):
        return [(f"E{i}", i, e) for i, e in enumerate(ents) if e["cond"] != "always"]

    def removal_view(self, ents, i):
        return [e for j, e in enumerate(ents) if j != i]

    def build(self):
        run, rows = self.run, self.run.rows
        marker = os.path.join(self.out, "export_summary.json")
        if os.path.exists(marker):
            os.replace(marker, marker + ".previous")
        allr = sorted(rows)
        ents = self.system()
        tip_s = range(int(self.cfg["tip_samples"]))
        stock_s, floor_s = range(int(self.cfg["stock_samples"])), range(int(self.cfg["floor_samples"]))
        rem_s = range(int(self.cfg["removal_samples"]))
        run.routes_for(ents, allr, "export")
        units = self.units(ents)
        routed = {g: [r for r in allr if run.fires(e, r)] for g, i, e in units}
        missing = dict(tip=self.request("tip", ents, allr, tip_s), stock=self.request("stock", [], allr, stock_s), floor=self.request("floor", self.base_entries(), allr, floor_s))
        for g, i, e in units:
            missing[f"wo_{g}"] = self.request(f"wo_{g}", self.removal_view(ents, i), routed[g], rem_s, self.removal_salt(g))
        out, keys_of_file, status = {}, {}, {}
        role = {r: rows[r][ROLE_FIELD] for r in allr}
        tail = int(self.cfg["tail_chars"])

        K = self.keys_of(ents, allr, tip_s)
        got = scan(run.io.root, set(K), status)
        arm = self.spec["tip_reads"]["arm"]
        L = [read_line(rows, arm, got[k], *K[k], key=k, role=role[K[k][0]], tail=(got[k].get("reasoning") or "")[-tail:]) for k in sorted(K, key=lambda k: K[k]) if k in got]
        out["tip_reads"] = write(os.path.join(self.out, self.spec["tip_reads"]["file"]), L)
        keys_of_file["tip_reads"] = set(K)

        head = dict(tip=run.tip()["id"], system=self.cfg["system"],
                    entries=[dict(i=i, cid=e.get("cid"), cond=e["cond"], condition=e.get("condition"), procedure=e.get("procedure"), settings=e.get("settings")) for i, e in enumerate(ents)])
        L = [head] + [dict(row=r, bench=rows[r]["bench"], role=role[r], fired=[i for i, e in enumerate(ents) if run.fires(e, r)]) for r in allr]
        out["tip_fired"] = write(os.path.join(self.out, self.spec["tip_fired"]["file"]), L)
        keys_of_file["tip_fired"] = set()

        L = []
        keys_of_file["entry_removals"] = set()
        for g, i, e in units:
            K = self.keys_of(self.removal_view(ents, i), routed[g], rem_s, self.removal_salt(g))
            got = scan(run.io.root, set(K), status)
            keys_of_file["entry_removals"] |= set(K)
            L += [read_line(rows, f"lrn_wo_{g}", got[k], *K[k], key=k) for k in sorted(K, key=lambda k: K[k]) if k in got]
            L.append(dict(arm=f"rows_wo_{g}", rows=routed[g]))
        out["entry_removals"] = write(os.path.join(self.out, self.spec["entry_removals"]["file"]), L)

        L = [dict(entry=i, cid=e.get("cid"), card_source=(e.get("source") or [None])[0], source=list(e.get("source") or []), author=e.get("author"),
                  layer="specialisation" if e in run.tip()["special"] else "general") for i, e in enumerate(ents)]
        out["entry_sources"] = write(os.path.join(self.out, self.spec["entry_sources"]["file"]), L)
        keys_of_file["entry_sources"] = set()

        fs = self.spec["floor_and_stock"]
        Ks = self.keys_of([], allr, stock_s)
        Kf = self.keys_of(self.base_entries(), allr, floor_s)
        gs, gf = scan(run.io.root, set(Ks), status), scan(run.io.root, set(Kf), status)
        L = [read_line(rows, fs["floor_arm"], gf[k], *Kf[k], key=k, role=role[Kf[k][0]]) for k in sorted(Kf, key=lambda k: Kf[k]) if k in gf]
        L += [read_line(rows, fs["stock_arm"], gs[k], *Ks[k], key=k, role=role[Ks[k][0]]) for k in sorted(Ks, key=lambda k: Ks[k]) if k in gs]
        out["floor_and_stock"] = write(os.path.join(self.out, fs["file"]), L)
        keys_of_file["floor_and_stock"] = set(Kf) | set(Ks)
        L = [read_line(rows, self.spec["floor_full"]["floor_arm"], gf[k], *Kf[k], key=k, role=role[Kf[k][0]], tail=(gf[k].get("reasoning") or "")[-tail:])
             for k in sorted(Kf, key=lambda k: Kf[k]) if k in gf]
        out["floor_full"] = write(os.path.join(self.out, self.spec["floor_full"]["file"]), L)
        keys_of_file["floor_full"] = set(Kf)

        by_file = {self.spec[k]["file"]: count_reads(ks, status) for k, ks in keys_of_file.items()}
        total = count_reads(set().union(*keys_of_file.values()), status)
        summary = dict(tip=run.tip()["id"], system=self.cfg["system"], entries=len(ents), units=[g for g, _, _ in units],
                       export_settings_sha256=FP.digest(FP.canonical(self.cfg)), read_settings_sha256=run.origin["read_settings_sha256"], read_settings_form="queue", totals_over_distinct_keys=True, expected_reads=total["expected"], missing_reads=total["missing_reads"], errored_reads=total["errored_reads"],
                       skipped_reads=total["skipped_reads"], reads_by_file_keys_overlap=by_file, unanswered_by_request=missing, base=self.base_arm(), files=out, complete=True)
        with open(marker + ".tmp", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        os.replace(marker + ".tmp", marker)
        run.log(f"  judge exports in {self.out}: " + ", ".join(f"{k} {v['lines']}" for k, v in out.items()) + f"; reads missing {total['missing_reads']}, errored {total['errored_reads']}, skipped {total['skipped_reads']} (over distinct keys)"
                + ("; to make the errored reads again, restart the dispatcher, then resume the loop once: it requests them again and waits for them before it writes the export" if total["errored_reads"] else ""))
        return summary
