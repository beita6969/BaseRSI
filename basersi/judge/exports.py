import gzip
import hashlib
import json
import os

from basersi import config
from basersi.engine import grader as G
from basersi.judge import frontier as P
from basersi.judge import guess as GU
from basersi.judge import reads as R

REMOVAL_ARM_PREFIX = "lrn_wo_"
REMOVAL_ROWS_PREFIX = "rows_wo_"


class ExportIncomplete(config.Refused, RuntimeError):
    pass


class ExportStale(ExportIncomplete):
    pass


ACCOUNTING = ("expected_reads", "missing_reads", "errored_reads", "skipped_reads", "reads_by_file_keys_overlap")
REBUILD = "Rebuild the export by resuming the loop on its run folder (it rewrites the export from the cached reads)"


def accounted(summary, where):
    lacking = [k for k in ACCOUNTING if k not in summary]
    if summary.get("totals_over_distinct_keys") is not True or lacking:
        raise ExportIncomplete(f"{where} has no read accounting (totals_over_distinct_keys: true and {', '.join(ACCOUNTING)}; lacking: "
                               f"{', '.join(lacking) or 'totals_over_distinct_keys'}), so the judge cannot tell whether reads are missing or errored. {REBUILD}")
    return summary


def graded_alike(summary, where):
    have, form, now = summary.get("read_settings_sha256"), summary.get("read_settings_form"), config.read_settings_sha256()
    if form != "queue" or have != now:
        raise ExportStale(f"{where} was read and graded under {'other read settings' if have else 'read settings it does not record'} "
                          f"(read_settings_sha256 {str(have)[:12]}, form {form}; now {now[:12]}, form queue: engine.yaml read settings and benchmarks.yaml answer_match): "
                          f"each read's ok was graded under those rules, so judging it under the current ones would mix the two. Restore the settings it was made under, "
                          f"or make the export again under the current ones: a new loop run folder (loop.run_name), since a run folder is never continued under other read settings")
    return summary


def _whole(x, what):
    if x is None:
        return 0
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise ExportIncomplete(f"the export summary gives {what} = {x!r}; it must be a whole number of reads. {REBUILD}")
    return x


def _counts(v, where="the totals"):
    skipped = v.get("skipped_reads") or {}
    if not isinstance(skipped, dict):
        raise ExportIncomplete(f"the export summary gives skipped_reads of {where} = {skipped!r}; it must map each cause to a number of reads. {REBUILD}")
    return (_whole(v.get("missing_reads"), f"missing_reads of {where}"), _whole(v.get("errored_reads"), f"errored_reads of {where}"),
            {str(k): n for k, n in ((k, _whole(n, f"skipped_reads.{k} of {where}")) for k, n in skipped.items()) if n})


def reads_account(summary):
    missing, errored, skipped = _counts(summary)
    by_file = summary.get("reads_by_file_keys_overlap") or {}
    if not isinstance(by_file, dict) or not all(isinstance(v, dict) for v in by_file.values()):
        raise ExportIncomplete(f"the export summary gives reads_by_file_keys_overlap in another form than one mapping of counts per file. {REBUILD}")
    lacking, broken = {}, []
    sums = dict(missing=0, errored=0, skipped={})
    for f, v in by_file.items():
        m, e, s = _counts(v, f)
        sums["missing"] += m
        sums["errored"] += e
        for k, n in s.items():
            sums["skipped"][k] = sums["skipped"].get(k, 0) + n
        if m > missing or e > errored or any(n > skipped.get(k, 0) for k, n in s.items()):
            broken.append(f"{f} counts more than the totals (missing {m}, errored {e}, skipped {s})")
        if v.get("expected") is not None and m + e + sum(s.values()) > _whole(v["expected"], f"expected of {f}"):
            broken.append(f"{f} counts more unread keys than it expects ({v['expected']})")
        if m or e:
            lacking[f] = dict(missing=m, errored=e)
    if by_file and (missing > sums["missing"] or errored > sums["errored"] or any(n > sums["skipped"].get(k, 0) for k, n in skipped.items())):
        broken.append(f"the totals (missing {missing}, errored {errored}, skipped {skipped}) exceed the sums over the files {sums}")
    if summary.get("expected_reads") is not None and missing + errored + sum(skipped.values()) > _whole(summary["expected_reads"], "expected_reads"):
        broken.append(f"the totals count more unread keys than the export expects ({summary['expected_reads']})")
    if broken:
        raise ExportIncomplete("the export summary breaks the counting rules (totals over distinct keys; a file's count is at most the total of its kind; "
                               "the totals are at most the sums over the files): " + "; ".join(broken) + ". " + REBUILD)
    return dict(missing_reads=missing, errored_reads=errored, skipped_reads=skipped, files_lacking=lacking)


class Exports:
    def __init__(self, export_dir, spec, sha256=None):
        self.dir = export_dir
        self.spec = spec
        self.sha256 = dict(sha256 or {})
        self.reads = None
        self.summary = self.check_complete()

    def check_complete(self):
        if not isinstance(self.spec.get("summary"), dict) or not self.spec["summary"].get("file"):
            raise config.ConfigError(f"judge exports.summary is {self.spec.get('summary')!r}: every judged export is checked against its summary file "
                                     f"(completeness, read accounting, read settings), so it must name that file (shipped: export_summary.json)")
        name = self.spec["summary"]["file"]
        p = self.spec["summary"].get("path") or os.path.join(self.dir, name)
        if not os.path.exists(p):
            raise ExportIncomplete(f"the export in {self.dir} is incomplete: {name} is missing (the loop writes it last, after every export file); "
                                   f"finish or resume the export, then run the judge")
        with open(p, encoding="utf-8") as f:
            summary = json.load(f)
        if summary.get("complete") is False:
            raise ExportIncomplete(f"the export in {self.dir} is incomplete: {name} says complete = false")
        missing = [k for k, v in (summary.get("files") or {}).items() if not os.path.exists(os.path.join(self.dir, os.path.basename(v["path"])))]
        if missing:
            raise ExportIncomplete(f"the export in {self.dir} is incomplete: {name} lists files that are not in the folder: {missing}")
        self.reads = reads_account(accounted(summary, f"the export in {self.dir} ({name})"))
        graded_alike(summary, f"the export in {self.dir} ({name})")
        self.reads["allowed"] = bool(self.spec.get("allow_missing_reads"))
        r = self.reads
        if (r["missing_reads"] or r["errored_reads"]) and not r["allowed"]:
            raise ExportIncomplete(f"the export in {self.dir} lacks reads ({name}, over distinct reads: {r['missing_reads']} missing, {r['errored_reads']} errored"
                                   f"{'; per file, files may share reads: ' + json.dumps(r['files_lacking']) if r['files_lacking'] else ''}): the judge would score those rows on fewer reads. "
                                   f"Missing reads: resume the loop (same run folder) with the servers and the dispatcher running. Errored reads: restart the dispatcher "
                                   f"(basersi serve dispatcher), then resume the loop once: it requests them again and waits for them before it rewrites the export. Or set exports.allow_missing_reads: true in the judge config "
                                   f"to judge it anyway (the verdict files then say so)")
        return summary

    def reads_lines(self):
        r = getattr(self, "reads", None)
        if not r:
            return []
        out = []
        if r["missing_reads"] or r["errored_reads"]:
            out.append(f"EXPORT READS: {r['missing_reads']} missing and {r['errored_reads']} errored reads, judged anyway because exports.allow_missing_reads is true; "
                       f"the rows concerned are scored on fewer reads; per file (files may share reads): {json.dumps(r['files_lacking'])}")
        if r["skipped_reads"]:
            out.append(f"export reads skipped by design (not missing; count by cause): {json.dumps(r['skipped_reads'], sort_keys=True)}")
        return out

    def base(self):
        given = (self.summary or {}).get("base")
        if given:
            out = dict(given)
            out.setdefault("arm", self.spec["floor_and_stock"]["floor_arm"])
            out.setdefault("entries", [])
            out.setdefault("text_entries", 0)
            out["settings"] = dict(out.get("settings") or {})
            return out
        spec = self.spec["floor_and_stock"]
        if spec.get("base_settings") is None:
            raise ExportIncomplete(f"the export in {self.dir} does not name its base arm: no 'base' in its summary and no exports.floor_and_stock.base_settings in the judge config")
        return dict(arm=spec["floor_arm"], settings=dict(spec["base_settings"]), entries=[], text_entries=int(spec.get("base_text_entries") or 0), origin="judge_config")

    def path(self, key):
        return self.spec[key].get("path") or os.path.join(self.dir, self.spec[key]["file"])

    def exists(self, key):
        return key in self.spec and os.path.exists(self.path(key))

    def lines(self, key):
        name = self.spec[key]["file"]
        p = self.path(key)
        if name in self.sha256:
            sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
            if sha != self.sha256[name]:
                raise ExportIncomplete(f"{name}: sha256 {sha[:16]}... is not the expected {self.sha256[name][:16]}...")
        return [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8") if l.strip()]

    def tip(self, rows_):
        L1 = self.lines("tip_reads")
        arm_name = self.spec["tip_reads"]["arm"]
        if self.exists("tip_reads_fix"):
            fix = {v["key"]: v for v in self.lines("tip_reads_fix")}
            L1 = [dict(fix[v["key"]], arm=arm_name) if (v.get("finish") == "abort" and v["key"] in fix) else v for v in L1]
        tip = arms_of(L1, rows_)[arm_name]
        L = self.lines("tip_fired")
        ents = L[0]["entries"]
        fired = {v["row"]: list(v["fired"]) for v in L[1:]}
        return tip, fired, ents

    def floor_and_stock(self, rows_):
        spec = self.spec["floor_and_stock"]
        given = ((self.summary or {}).get("base") or {}).get("arm")
        a = arms_of(self.lines("floor_and_stock"), rows_)
        floor = a[given or spec["floor_arm"]]
        if self.exists("floor_full"):
            floor = arms_of(self.lines("floor_full"), rows_)[given or self.spec["floor_full"]["floor_arm"]]
        return floor, a[spec["stock_arm"]]

    def entry_removals(self, rows_):
        L = self.lines("entry_removals")
        arms = arms_of([v for v in L if v.get("arm", "").startswith(REMOVAL_ARM_PREFIX)], rows_)
        wo = {a[len(REMOVAL_ARM_PREFIX):]: t for a, t in arms.items()}
        routed = {v["arm"][len(REMOVAL_ROWS_PREFIX):]: list(v["rows"]) for v in L if v.get("arm", "").startswith(REMOVAL_ROWS_PREFIX)}
        return wo, routed

    def entry_sources(self):
        src = {}
        for v in self.lines("entry_sources"):
            given = v["source"] if isinstance(v.get("source"), list) else [v.get("card_source")]
            src.setdefault(int(v["entry"]), set()).update(b for b in given if b)
        return {i: sorted(s) for i, s in src.items()}


def settings_text(settings):
    return ", ".join(f"{k} {v}" for k, v in settings.items()) or "stock settings"


def base_label(base):
    st = settings_text(base["settings"])
    n = int(base.get("text_entries") or 0)
    if n == 0:
        return f"floor-only settings ({st}, no text entry; cached reads on every development row)", f"image floor {st}"
    return (f"the tip's general part ({st}, {n} text entr{'y' if n == 1 else 'ies'}; cached reads on every development row)",
            f"general part {st} + {n} text entr{'y' if n == 1 else 'ies'}")


def read_of(v, rows_):
    if v["row"] not in rows_:
        raise R.RowError(f"export row {v['row']} is not a development row")
    if v.get("finish") == "abort" or v.get("error") or v.get("ok") is None:
        return None
    c = v.get("cls")
    return dict(ok=bool(v["ok"]), tokens=float(v.get("tokens") or 0.0), pred=v.get("pred"), cls=(P.NONE if c in (None, "'none'", "none") else c),
                guess=None, finish=v.get("finish"), s=int(v["s"]), prompt_tokens=float(v.get("prompt_tokens") or 0.0), tail=v.get("tail"), key=v.get("key"))


def arms_of(lines, rows_):
    arms = {}
    for v in lines:
        if "row" not in v or "s" not in v:
            continue
        x = read_of(v, rows_)
        if x is not None:
            arms.setdefault(v["arm"], {}).setdefault(v["row"], {}).setdefault(x["s"], x)
    return {a: {r: [d[s] for s in sorted(d)] for r, d in t.items()} for a, t in arms.items()}


def full_rows(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            out[r["id"]] = r
    return out


def classify(arm, full):
    for r, xs in arm.items():
        row = full[r]
        for x in xs:
            p = x.get("pred")
            if p in ("", "None"):
                p = None
            if p is not None and row.get("kind") == "num" and isinstance(p, str):
                try:
                    p = float(p)
                except ValueError:
                    pass
            c = G.answer_class(p, row)
            x["cls"] = P.NONE if c == G.NO_ANSWER_CLASS else c
    return arm


def set_guess(arm, det, windows=None):
    ws = windows or [det["window"]]
    for xs in arm.values():
        for x in xs:
            x["guess"] = {w: GU.flag(dict(det, window=w), x.get("tail")) for w in ws}
    return arm
