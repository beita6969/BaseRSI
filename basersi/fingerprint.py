import contextlib
import functools
import hashlib
import json
import math
import os
import shlex
import sys
import time

from basersi import config

SIDECAR = ".inputs.json"
MARKER = "basersi_edit.json"
MODEL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
               "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt", "preprocessor_config.json", "processor_config.json",
               "video_preprocessor_config.json", "chat_template.json", "chat_template.jinja", MARKER)
MODEL_FILE_SUFFIXES = (".safetensors", ".safetensors.index.json", ".bin", ".bin.index.json")
_FILE_DIGESTS = {}
OTHER_CODE_KEY = "resume.accept_outputs_of_other_code"
_POLICY = dict(other_code=False)


class StaleOutput(config.Refused, RuntimeError):
    pass


class InputRefused(config.Refused, ValueError):
    pass


class OutputExists(config.Refused, FileExistsError):
    pass


def digest(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def canonical(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


ROWS_MANIFEST = "rows_manifest_sha256"


@functools.lru_cache(maxsize=None)
def code():
    return config.code_identity()


@functools.lru_cache(maxsize=None)
def rows_manifest():
    return config.rows_manifest_sha256()


def code_of(reads_rows=True):
    return dict(code())


def with_code(inputs, reads_rows=True):
    out = dict(inputs, code=code_of())
    if reads_rows:
        out[ROWS_MANIFEST] = rows_manifest()
    return canonical(out)


def fingerprint(inputs, reads_rows=True):
    return digest(with_code(inputs, reads_rows))


@contextlib.contextmanager
def other_code(allowed):
    before = _POLICY["other_code"]
    _POLICY["other_code"] = bool(allowed)
    try:
        yield
    finally:
        _POLICY["other_code"] = before


def only_code_differs(recorded_inputs, inputs, reads_rows=True):
    return recorded_inputs is not None and differing(recorded_inputs, with_code(inputs, reads_rows)) == ["code"]


def reused_with_other_code(path, recorded_inputs, inputs, reads_rows=True):
    if not (_POLICY["other_code"] and only_code_differs(recorded_inputs, inputs, reads_rows)):
        return None
    made_by = (recorded_inputs or {}).get("code") or {}
    print(f"[basersi] reusing {path}: made by other package code (version {made_by.get('version')}, sources {str(made_by.get('sources_sha256'))[:12]}) "
          f"with the same inputs, because {OTHER_CODE_KEY} is true", file=sys.stderr, flush=True)
    return dict(recorded=made_by, now=code_of(reads_rows), key=OTHER_CODE_KEY, at=time.strftime("%Y-%m-%d %H:%M:%S %z"))


def match(recorded_fingerprint, recorded_inputs, inputs, path, reads_rows=True):
    if recorded_fingerprint == fingerprint(inputs, reads_rows):
        return "same"
    return reused_with_other_code(path, recorded_inputs, inputs, reads_rows)


REUSE_RECORD = "reused_with_other_code"


def with_other_code_record(record, matched):
    if not isinstance(matched, dict):
        return record
    have = list(record.get(REUSE_RECORD) or [])
    pair = digest([canonical(matched["recorded"]), canonical(matched["now"])])
    if any(digest([canonical(r.get("recorded")), canonical(r.get("now"))]) == pair for r in have):
        return record
    return dict(record, **{REUSE_RECORD: have + [matched]})


def reuse_records_of(report_path):
    have = read_json(report_path) if os.path.isfile(report_path) else {}
    return {REUSE_RECORD: have[REUSE_RECORD]} if have.get(REUSE_RECORD) else {}


def without_reuse_record(record):
    return {k: v for k, v in record.items() if k != REUSE_RECORD} if isinstance(record, dict) else record


SPEC_OPS = ("set", "scale", "add")
SPEC_EXPECTED = "a Procedure Edit spec (a JSON list of ops, each a mapping with op set / scale / add, param and neurons)"


def given_path(path, arg, kind="file"):
    exists = os.path.isdir if kind == "folder" else os.path.isfile
    if not isinstance(path, str) or not path or not exists(path):
        what = "does not exist" if not (isinstance(path, str) and path and os.path.lexists(path)) else f"is not a {kind}"
        raise config.MissingInput(f"{arg} {path}: {what}; expected an existing {kind}")
    return path


def given_json(path, arg, expected, valid=None):
    given_path(path, arg, "file")
    try:
        value = read_json(path)
    except (UnicodeDecodeError, ValueError) as e:
        raise InputRefused(f"{arg} {path} is not a JSON file ({type(e).__name__}: {e}); expected {expected}") from None
    if valid is not None and not valid(value):
        raise InputRefused(f"{arg} {path} is not {expected}")
    return value


def is_spec(value):
    return (isinstance(value, list) and bool(value)
            and all(isinstance(op, dict) and op.get("op") in SPEC_OPS and isinstance(op.get("param"), str)
                    and isinstance(op.get("sel", op), dict) and "neurons" in op.get("sel", op) for op in value))


def given_checkpoint(path, arg):
    given_path(path, arg, "folder")
    index = os.path.join(path, "model.safetensors.index.json")
    if not os.path.isfile(index):
        raise InputRefused(f"{arg} {path} is not a checkpoint folder (it holds no model.safetensors.index.json); expected a checkpoint folder")
    given_json(index, f"{arg} index", "a safetensors index (a JSON object with a weight_map object)",
               lambda v: isinstance(v, dict) and isinstance(v.get("weight_map"), dict))
    return path


def file_sha256(path):
    st = os.stat(path)
    cache_key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    if cache_key not in _FILE_DIGESTS:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        _FILE_DIGESTS[cache_key] = h.hexdigest()
    return _FILE_DIGESTS[cache_key]


def model_files(folder):
    names = []
    for name in sorted(os.listdir(folder)):
        if name.startswith(".") or not os.path.isfile(os.path.join(folder, name)):
            continue
        if name in MODEL_FILES or name.endswith(MODEL_FILE_SUFFIXES):
            names.append(name)
    return names


def checkpoint_identity(folder):
    files = {}
    for name in model_files(folder):
        if name == MARKER:
            marker = read_json(os.path.join(folder, name))
            files[name] = dict(content_sha256=marker.get("content_sha256"), pending=marker.get("pending"))
        else:
            files[name] = file_sha256(os.path.join(folder, name))
    return digest(files)


def sidecar(path):
    return path.rstrip("/") + SIDECAR


def recorded(path):
    p = sidecar(path)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path, data, indent=None):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    os.replace(tmp, path)
    return path


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aside_name(path):
    path = path.rstrip("/")
    n = 1
    while os.path.lexists(f"{path}.set-aside-{n}"):
        n += 1
    return f"{path}.set-aside-{n}"


def set_aside(path):
    target = aside_name(path)
    os.rename(path.rstrip("/"), target)
    print(f"[basersi] moved {path} (left by an interrupted run) to {target}; it is not used again and can be removed by hand "
          f"(a checkpoint copy takes the full model size on disk)", file=sys.stderr, flush=True)
    return target


def move_commands(paths):
    return "; ".join(f"mv {shlex.quote(p)} {shlex.quote(aside_name(p))}" for p in paths if os.path.lexists(p))


def way_forward(paths, key=None):
    options = [f"use a new work_dir{f' or set {key} to a new path' if key else ''}"]
    moves = move_commands(paths)
    if moves:
        options.append(f"or move the old files away: {moves}")
    return ", ".join(options)


def differing(old, new):
    old, new = canonical(old or {}), canonical(new)
    keys = sorted(set(old) | set(new))
    return [k for k in keys if digest(old.get(k)) != digest(new.get(k))]


def stale_message(path, have, inputs, records=None, key=None, reads_rows=True):
    if have is None:
        why = "has no record of the inputs that made it"
    else:
        why = "was made with other inputs (differ: " + ", ".join(differing(have.get("inputs"), with_code(inputs, reads_rows))) + ")"
    files = [path] + list(records if records is not None else [sidecar(path)])
    if have is not None and ROWS_MANIFEST in differing(have.get("inputs"), with_code(inputs, reads_rows)):
        return (f"{path} {why}; the shipped row manifest changed, which is a rows change: it is never reused, not even with {OTHER_CODE_KEY}; "
                f"{way_forward(files, key)}")
    if have is not None and only_code_differs(have.get("inputs"), inputs, reads_rows):
        return (f"{path} {why}; only the package code differs: set {OTHER_CODE_KEY} to reuse it knowingly (the reuse is recorded and printed), "
                f"or {way_forward(files, key)}")
    return f"{path} {why}; nothing is reused across a change: {way_forward(files, key)}"


def reuse(path, inputs, key=None, reads_rows=True):
    value = fingerprint(inputs, reads_rows)
    have = recorded(path)
    if os.path.exists(path):
        if have is not None and have.get("fingerprint") == value:
            return True
        reused = reused_with_other_code(path, have.get("inputs"), inputs, reads_rows) if have is not None else None
        if reused:
            updated = with_other_code_record(have, reused)
            if updated is not have:
                write_json_atomic(sidecar(path), updated, indent=1)
            return True
        raise StaleOutput(stale_message(path, have, inputs, key=key, reads_rows=reads_rows))
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    write_json_atomic(sidecar(path), dict(fingerprint=value, inputs=with_code(inputs, reads_rows)), indent=1)
    return False


def of(path):
    have = recorded(path)
    if have is None or not os.path.exists(path):
        raise StaleOutput(f"{path} has no record of the inputs that made it")
    return have["fingerprint"]


def read_marker(folder):
    p = os.path.join(folder, MARKER)
    return read_json(p) if os.path.exists(p) else None


def marker_edits(folder):
    marker = read_marker(folder)
    if marker is None:
        return []
    if marker.get("pending"):
        raise StaleOutput(f"{folder} was being written when the run stopped ({MARKER} names a pending {marker['pending'].get('stage')}): "
                          f"it is neither the old nor the new checkpoint; edit a fresh copy instead (basersi export checkpoint --out NEW_FOLDER), "
                          f"or move it away: {move_commands([folder])}")
    return list(marker["edits"])


def edit_keys(edits):
    return [[e["stage"], e["inputs_fingerprint"]] for e in edits]


def require_stock(folder, role):
    marker = read_marker(folder)
    if marker is None or not (marker.get("pending") or marker.get("edits")):
        return
    from basersi.model_check import BaseModelError
    edits = marker.get("edits") or []
    first = edits[0]["report"] if edits else {}
    source = first.get("source") or first.get("reference")
    raise BaseModelError(f"{folder} holds {MARKER} ({', '.join(e['stage'] for e in edits) or 'pending'}): an edited checkpoint is never a stock base; "
                         f"{role} takes the stock folder{f' ({source})' if source else ''}")


def content_sha256(edits):
    return digest(edit_keys(edits))


def write_marker(folder, edits, pending=None):
    data = dict(edits=list(edits), pending=pending, content_sha256=None if pending else content_sha256(edits))
    write_json_atomic(os.path.join(folder, MARKER), data, indent=1)
    return data["content_sha256"]


def last_edit(folder, stage):
    marker = read_marker(folder)
    if not marker or marker.get("pending") or not marker.get("edits"):
        return None
    edit = marker["edits"][-1]
    return edit if edit.get("stage") == stage else None
