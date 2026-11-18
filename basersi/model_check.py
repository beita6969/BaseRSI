import copy
import hashlib
import os
import sys

from basersi import config
from basersi import fingerprint as FP

REQUIRED = ("id", "revision", "config_sha256")
WEIGHTS_CHECKS = ("quick", "full")
INDEX_FILE = "model.safetensors.index.json"
HEAD_BYTES = 1 << 20
_CHECKED = {}


class BaseModelError(config.Refused, ValueError):
    pass


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    left = limit
    with open(path, "rb") as f:
        while left is None or left > 0:
            block = f.read(1 << 24 if left is None else min(1 << 24, left))
            if not block:
                break
            h.update(block)
            if left is not None:
                left -= len(block)
    return h.hexdigest()


def download_revision(model_dir):
    p = os.path.join(model_dir, ".cache", "huggingface", "download", "config.json.metadata")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return f.readline().strip() or None


def check_weights(model_dir, base_model, level, want):
    recorded = base_model.get("weights") or {}
    if not recorded:
        return dict(weights_checked=None), "base_model names no weights (shard names, sizes and hashes), so the weights were not checked"
    found = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    lacking, extra = sorted(set(recorded) - set(found)), sorted(set(found) - set(recorded))
    if lacking or extra:
        raise BaseModelError(f"{model_dir}: the weight shards are not those of {want}: missing {lacking or 'none'}, not expected {extra or 'none'}")
    for name in sorted(recorded):
        p = os.path.join(model_dir, name)
        size = os.path.getsize(p)
        if size != int(recorded[name]["size"]):
            raise BaseModelError(f"{p}: {size} bytes, the shard of {want} has {recorded[name]['size']}; download it again")
        key, got = ("first_mib_sha256", sha256_file(p, HEAD_BYTES)) if level == "quick" else ("sha256", sha256_file(p))
        if got != recorded[name][key]:
            raise BaseModelError(f"{p}: {'the sha256 of its first MiB' if level == 'quick' else 'its sha256'} {got[:16]}... is not that of {want} ({recorded[name][key][:16]}...); download it again")
    idx = os.path.join(model_dir, INDEX_FILE)
    if base_model.get("index_sha256"):
        if not os.path.exists(idx):
            raise BaseModelError(f"{model_dir} has no {INDEX_FILE}; {want} has one")
        got = sha256_file(idx)
        if got != base_model["index_sha256"]:
            raise BaseModelError(f"{idx}: sha256 {got[:16]}... is not that of {want} ({base_model['index_sha256'][:16]}...)")
    what = "names, sizes and the sha256 of the first MiB" if level == "quick" else "names, sizes and the sha256 of every byte"
    return dict(weights_checked=level, weight_shards=len(recorded), weights_check=what), None


def folder_state(model_dir):
    out = []
    for name in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p) and (name in FP.MODEL_FILES or name.endswith(FP.MODEL_FILE_SUFFIXES)):
            st = os.stat(p)
            out.append((name, st.st_size, st.st_mtime_ns))
    meta = os.path.join(model_dir, ".cache", "huggingface", "download", "config.json.metadata")
    if os.path.exists(meta):
        st = os.stat(meta)
        out.append((meta, st.st_size, st.st_mtime_ns))
    return out


def check_base_model(model_dir, base_model, level=None):
    if model_dir and os.path.isdir(model_dir):
        key = (os.path.realpath(model_dir), FP.digest(FP.canonical(base_model or {})), level, FP.digest(folder_state(model_dir)))
        if key in _CHECKED:
            return copy.deepcopy(_CHECKED[key])
        out = _check(model_dir, base_model, level)
        _CHECKED[key] = copy.deepcopy(out)
        return out
    return _check(model_dir, base_model, level)


def _check(model_dir, base_model, level=None):
    missing = [k for k in REQUIRED if not (base_model or {}).get(k)]
    if missing:
        raise BaseModelError(f"base_model must give {list(REQUIRED)}; missing {missing}")
    level = level or base_model.get("weights_check") or "quick"
    if level not in WEIGHTS_CHECKS:
        raise BaseModelError(f"base_model.weights_check must be one of {list(WEIGHTS_CHECKS)}, not {level!r}")
    want = f"{base_model['id']} at revision {base_model['revision']}"
    if not model_dir or not os.path.isdir(model_dir):
        raise BaseModelError(f"model folder {model_dir!r} does not exist; it must hold {want}")
    FP.require_stock(model_dir, "the base model (model)")
    cfg = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg):
        raise BaseModelError(f"{model_dir} has no config.json; it must hold {want}")
    sha = sha256_file(cfg)
    if sha != base_model["config_sha256"]:
        raise BaseModelError(f"{cfg}: sha256 {sha[:16]}... is not the config.json of {want} ({base_model['config_sha256'][:16]}...); "
                             f"download {base_model['id']} with revision {base_model['revision']}")
    not_checked = []
    rev = download_revision(model_dir)
    if rev is not None and rev != base_model["revision"]:
        raise BaseModelError(f"{model_dir} was downloaded at revision {rev}, not {base_model['revision']} ({base_model['id']})")
    weights, note = check_weights(model_dir, base_model, level, want)
    if rev is None:
        not_checked.append("the folder has no Hugging Face download metadata (.cache/huggingface/download/config.json.metadata), so the revision was not checked"
                           f" (the config.json was{', and the weights: ' + weights['weights_check'] if weights.get('weights_checked') else ''})")
    if note:
        not_checked.append(note)
    if weights.get("weights_checked") == "quick":
        not_checked.append("weights_check quick proves the shard names, sizes and first MiB only; weights_check: full proves every byte")
    for n in not_checked:
        print(f"[basersi] base model check of {model_dir}: {n}", file=sys.stderr, flush=True)
    return dict(model_dir=os.path.realpath(model_dir), id=base_model["id"], revision=base_model["revision"], config_sha256=sha,
                revision_in_download_metadata=rev, revision_checked=rev is not None, **weights, not_checked=not_checked)
