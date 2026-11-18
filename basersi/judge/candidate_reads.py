import gzip
import json
import os
import sys

from basersi import config
from basersi import fingerprint as FP
from basersi.engine import reads as ER
from basersi.judge import reads as R

TAIL_CHARS = int(config.load("loop")["judge_export"]["tail_chars"])


def export_line(arm, x, weights=None):
    d = dict(arm=arm, row=x["row"], s=int(x["sample"]), ok=bool(x.get("ok")), pred=x.get("pred"), tokens=x.get("tokens"),
             prompt_tokens=x.get("prompt_tokens"), finish=x.get("finish"), key=f"{arm}|{x['row']}|{int(x['sample'])}",
             image_tokens=x.get("image_tokens"), tail=(x.get("reasoning") or "")[-TAIL_CHARS:])
    if weights is not None:
        d["weights"] = weights
    return d


def to_export(read_path, arm, out_gz, weights=None):
    lines = []
    seen = set()
    for l in open(read_path, encoding="utf-8"):
        if not l.strip():
            continue
        x = json.loads(l)
        if x.get("error"):
            continue
        k = (x["row"], int(x["sample"]))
        if k in seen:
            continue
        seen.add(k)
        lines.append(export_line(arm, x, weights))
    lines.sort(key=lambda d: (d["row"], d["s"]))
    os.makedirs(os.path.dirname(os.path.abspath(out_gz)), exist_ok=True)
    with gzip.open(out_gz, "wt", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return dict(path=out_gz, lines=len(lines))


def read_identity(model_dir, rows_path, roles, samples):
    return dict(checkpoint=FP.checkpoint_identity(os.path.realpath(model_dir)), rows_sha256=FP.file_sha256(rows_path), roles=list(roles), samples=list(samples),
                read_settings_sha256=config.read_settings_sha256(direct=True), code=config.code_identity(), rows_manifest_sha256=config.rows_manifest_sha256())


def recorded_identity(side):
    if not os.path.exists(side):
        return None, "it has no record of what it was read under"
    try:
        with open(side, encoding="utf-8") as f:
            return json.load(f), None
    except (OSError, ValueError) as e:
        return None, f"its record {side} cannot be read ({type(e).__name__})"


def fresh_reads(read_path, model_dir, rows_path, roles, samples, accept_other_code=False):
    side = read_path + ".source.json"
    want = FP.canonical(read_identity(model_dir, rows_path, roles, samples))
    if os.path.exists(read_path):
        have, why = recorded_identity(side)
        if accept_other_code and have is not None and FP.differing(have, want) == ["code"]:
            print(f"[basersi] {read_path}: read by other code ({have['code']}; now {want['code']}); reused because resume.accept_outputs_of_other_code is true",
                  file=sys.stderr, flush=True)
            return dict(recorded=have["code"], now=want["code"], key="resume.accept_outputs_of_other_code")
        if have is None or FP.digest(FP.canonical(have)) != FP.digest(want):
            why = why or "it was read under another " + ", ".join(k.replace("_sha256", "").replace("_", " ") for k in FP.differing(have, want))
            moved = R.superseded_name(read_path)
            os.replace(read_path, moved)
            if os.path.exists(side):
                os.replace(side, moved + ".source.json")
            print(f"[basersi] {read_path}: {why}; moved to {moved} and read again", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(read_path)), exist_ok=True)
    FP.write_json_atomic(side, want, indent=1)
    return None


def read_candidate(model_dir, rows_path, roles, samples, arm, read_path, out_gz, cards, port, weights_label=None, accept_other_code=False, allow_missing_reads=False):
    other = fresh_reads(read_path, model_dir, rows_path, roles, samples, accept_other_code)
    rows = R.load_rows(rows_path)
    row_ids = R.rows_of(rows, roles)
    ER.read_rows(model_dir, row_ids, samples, {"settings": {}}, read_path, rows_path=rows_path, port=port, cards=cards)
    out = to_export(read_path, arm, out_gz, weights=weights_label or os.path.basename(os.path.normpath(model_dir)))
    out["expected_reads"] = len(row_ids) * len(samples)
    out["missing_reads"] = out["expected_reads"] - out["lines"]
    if out["missing_reads"]:
        from basersi.judge.exports import ExportIncomplete
        msg = (f"{read_path}: {out['missing_reads']} of {out['expected_reads']} candidate reads are missing or errored after every retry, so "
               f"the rows concerned would be scored on fewer reads")
        if not allow_missing_reads:
            raise ExportIncomplete(msg + "; run the judge again (it reads only the missing ones), or set exports.allow_missing_reads: true to judge anyway")
        print(f"[basersi] {msg}; judged anyway because exports.allow_missing_reads is true", file=sys.stderr, flush=True)
    if other:
        out["read_by_other_code"] = other
    return out
