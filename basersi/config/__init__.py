import copy
import difflib
import hashlib
import json
import math
import os
import sys

import yaml

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = ("development_manifest.jsonl",)
FREE_FORM = {("benchmarks", "answer_match"), ("benchmarks", "id_stem_patterns"), ("sources", "datasets"), ("loop", "settings_menu"),
             ("pipeline", "base_model", "weights")}


class Refused(Exception):
    pass


class ConfigError(Refused, ValueError):
    pass


class MissingInput(Refused, FileNotFoundError):
    pass


EXIT_CODES = ("exit codes: 0 done; 2 refused (REFUSED: <reason> on stderr, no traceback: a config error, an input that is missing,\n"
              "  unreadable, stale, made by other code or not the expected model, rows, images or export reads\n"
              "  that fail their check), or a usage error reported by the argument parser (its usage line, no REFUSED line);\n"
              "  3 stopped at an acceptance gate; anything else stops with its traceback")


def need_file(file_path, what):
    if not os.path.isfile(file_path):
        raise MissingInput(f"{what} {file_path} does not exist")
    return file_path


def run_command(main, argv=None):
    try:
        rc = main(argv)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    return 0 if rc is None else int(rc)


def path(name):
    if name.endswith(".yaml"):
        name = name[:-5]
    return os.path.join(CONFIG_DIR, name + ".yaml")


def _read(file_path, what="config"):
    need_file(file_path, what)
    try:
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        raise ConfigError(f"{what} {file_path} is not readable yaml: {e}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config {file_path!r} must be a mapping, got {type(data).__name__}")
    return data


def read_json(file_path, what, kind=None):
    need_file(file_path, what)
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except (ValueError, UnicodeDecodeError) as e:
        raise ConfigError(f"{what} {file_path} is not readable json: {e}") from None
    if kind is not None and not isinstance(data, kind):
        raise ConfigError(f"{what} {file_path} must hold a json {'object' if kind is dict else 'list' if kind is list else kind.__name__}, got {type(data).__name__}")
    return data


def merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


VALUE_KEYS = {("judge", "entry_groups"): ("label", "entries"), ("pipeline", "base_model", "weights"): ("size", "first_mib_sha256", "sha256"),
              ("loop", "settings_menu"): ("stock", "values", "what")}
NOT_NULL = {("judge", "exports"), ("judge", "exports", "summary"), ("judge", "exports", "summary", "file")}
CHOICES = {("pipeline", "base_model", "weights_check"): ("quick", "full")}
PAIRS = {("procedure_edit", "teacher_samples"), ("procedure_edit", "base_floor_samples"), ("procedure_edit", "student", "samples")}


def shape(v):
    return "mapping" if isinstance(v, dict) else "list" if isinstance(v, list) else "scalar"


def kind(v):
    return "true / false" if isinstance(v, bool) else "number" if isinstance(v, (int, float)) else "text" if isinstance(v, str) else type(v).__name__


def is_pair(path):
    return tuple(path) in PAIRS or tuple(path[:-1]) + ("*",) in PAIRS


def problems(given, shipped, where):
    if not isinstance(given, dict):
        return []
    if tuple(where) in VALUE_KEYS:
        valid = VALUE_KEYS[tuple(where)]
        out = []
        for k, v in given.items():
            if isinstance(v, dict):
                out += [("unknown", list(where) + [k, x], sorted(valid)) for x in v if x not in valid]
            elif v is not None:
                out.append(("shape", list(where) + [k], (shape(v), dict.fromkeys(valid), [])))
        return out
    if not isinstance(shipped, dict) or tuple(where) in FREE_FORM:
        return []
    out = []
    for k, v in given.items():
        if k not in shipped:
            if shipped:
                out.append(("unknown", list(where) + [k], sorted(shipped)))
            continue
        d, here = shipped[k], list(where) + [k]
        if v is None and d is not None and tuple(here) in NOT_NULL:
            out.append(("null", here, d))
            continue
        if d is None or v is None:
            continue
        if shape(v) != shape(d):
            out.append(("shape", here, (shape(v), d, sorted(shipped))))
        elif isinstance(v, dict):
            out += problems(v, d, here)
        elif isinstance(v, list):
            items = [x for x in d if isinstance(x, dict)]
            if items:
                valid = sorted({k for x in items for k in x})
                for i, x in enumerate(v):
                    if isinstance(x, dict):
                        out += [("unknown", here + [f"[{i}]", y], valid) for y in x if y not in valid]
            if is_pair(here) and len(v) != len(d):
                out.append(("length", here, (len(v), d)))
            kinds = {kind(x) for x in d if x is not None}
            bad = [x for x in v if x is not None and kind(x) not in kinds]
            if len(kinds) == 1 and kinds <= {"number", "text", "true / false"} and bad:
                out.append(("type", here, (f"a list holding {kind(bad[0])} ({bad[0]!r})", d)))
        elif kind(v) != kind(d):
            out.append(("type", here, (f"{kind(v)} ({v!r})", d)))
        elif tuple(here) in CHOICES and v not in CHOICES[tuple(here)]:
            out.append(("choice", here, (v, CHOICES[tuple(here)])))
    return out


def hint(key, valid):
    close = difflib.get_close_matches(key, valid, n=2)
    return f" (did you mean {' or '.join(close)}?)" if close else ""


def valid_form(key, declared):
    if isinstance(declared, dict) and declared:
        return f"{key}: a mapping with the keys {', '.join(map(str, declared))}"
    text = yaml.safe_dump({key: declared}, default_flow_style=True, sort_keys=False, width=10 ** 6).strip()
    return text if len(text) <= 160 else text[:157] + "..."


def shape_hint(key, given, siblings):
    if isinstance(given, dict):
        alt = [f"{key}_{sub}" for sub in given if f"{key}_{sub}" in siblings]
        if alt:
            return f"; the settings under it go to {' and '.join(alt)}"
    return ""


def check_strength(cfg):
    strength = cfg["strength"]
    steps = strength["steps"]
    if not isinstance(steps, list) or not steps or any(isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0 for s in steps):
        raise ConfigError(f"procedure_edit.strength.steps must be a non-empty list of positive steps, got {steps!r}")
    if len({f"{float(s):g}" for s in steps}) != len(steps):
        raise ConfigError(f"procedure_edit.strength.steps holds a step twice (a step is named by its value to 6 significant digits), got {steps!r}")
    step = strength["step"]
    if step is not None and (isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0):
        raise ConfigError(f"procedure_edit.strength.step must be null (the line search over strength.steps) or a positive step, got {step!r}")


def check_keys(given, shipped, name, source=None, what=None, prefix=None):
    found = problems(given, shipped, [name])
    if not found:
        return
    label = lambda where: ".".join(str(x) for x in (prefix or []) + where[1:]).replace(".[", "[")
    unknown = [f"  {label(where)}{hint(where[-1], info)}; valid keys there: {', '.join(info)}" for k_, where, info in found if k_ == "unknown"]
    wrong = []
    for what_kind, where, info in found:
        key = str(where[-1])
        if what_kind == "shape":
            got, declared, siblings = info
            given_value = given
            for x in where[1:]:
                given_value = given_value.get(x) if isinstance(given_value, dict) else None
            wrong.append(f"  {label(where)}: a {got} was given where a {shape(declared)} is declared; the valid form is {valid_form(key, declared)}"
                         + shape_hint(key, given_value, siblings))
        elif what_kind == "type":
            got, declared = info
            want = f"a list of {kind(next(x for x in declared if x is not None))}" if isinstance(declared, list) else kind(declared)
            wrong.append(f"  {label(where)}: {got} was given where {want} is declared; the valid form is {valid_form(key, declared)}")
        elif what_kind == "choice":
            got, valid = info
            wrong.append(f"  {label(where)}: {got!r} is not one of {', '.join(valid)}{hint(str(got), list(valid))}")
        elif what_kind == "null":
            wrong.append(f"  {label(where)}: null was given where a value is required (it cannot be switched off); the valid form is {valid_form(key, info)}")
        elif what_kind == "length":
            got, declared = info
            wrong.append(f"  {label(where)}: {got} values were given where exactly {len(declared)} are declared; the valid form is {valid_form(key, declared)}")
    parts = []
    if unknown:
        parts.append(f"{source or 'the config'} has keys that {what or name + '.yaml'} does not declare (an unknown key would be silently ignored, so it is refused):\n"
                     + "\n".join(unknown))
    if wrong:
        parts.append(f"{source or 'the config'} has values whose shape differs from {what or name + '.yaml'} (mapping, list or scalar; number, text or true / false; the length of a pair), so they are refused:\n"
                     + "\n".join(wrong))
    raise ConfigError("\n".join(parts))


def load(name, override=None):
    data = _read(path(name))
    if override is None:
        return data
    source = None
    if isinstance(override, str):
        source = override
        override = _read(override, "the config file")
    check_keys(override, data, name, source)
    return merge(data, override)


def benchmarks():
    return list(load("benchmarks")["benchmarks"])


READ_SETTINGS_NOT_DEFINING = {
    "request_tries": None,
    "reads": ("tries", "timeout_seconds", "concurrency", "retry_wait_seconds"),
    "dispatcher": None,
    "serve": ("host", "ports", "start_wait_seconds", "stop_grace_seconds"),
}


LOOP_QUEUE_ONLY = ("chat_max_tokens", "chat_temperature", "router", "ops")


GRADING_KEYS = ("answer_match",)


def read_settings(engine=None, direct=False):
    e = copy.deepcopy(engine if engine is not None else load("engine"))
    bench = load("benchmarks")
    e["grading"] = {k: copy.deepcopy(bench.get(k) or {}) for k in GRADING_KEYS}
    if direct:
        for k in LOOP_QUEUE_ONLY:
            e.pop(k, None)
    for k, sub in READ_SETTINGS_NOT_DEFINING.items():
        if sub is None:
            e.pop(k, None)
        elif isinstance(e.get(k), dict):
            for s in sub:
                e[k].pop(s, None)
    return e


def read_settings_sha256(engine=None, direct=False):
    from basersi import fingerprint as FP
    return FP.digest(FP.canonical(read_settings(engine, direct)))


def _tree_sha256(root, suffixes):
    h = hashlib.sha256()
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            if f.endswith(suffixes):
                p = os.path.join(d, f)
                h.update(os.path.relpath(p, root).replace(os.sep, "/").encode() + b"\0")
                with open(p, "rb") as fh:
                    h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def code_identity():
    import basersi
    root = os.path.dirname(os.path.abspath(basersi.__file__))
    return dict(version=basersi.__version__, sources_sha256=_tree_sha256(root, (".py",)))


def rows_manifest_sha256():
    return _tree_sha256(CONFIG_DIR, DATA_FILES)
