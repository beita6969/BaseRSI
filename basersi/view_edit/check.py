import json
import os

import torch
from safetensors import safe_open

from basersi import fingerprint as fp
from basersi.view_edit.build import REWRITTEN
from basersi.view_edit.patch_kernel import PATCH_KERNEL


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def same_link(a, b):
    return os.path.realpath(a) == os.path.realpath(b)


def check_view_checkpoint(source, out):
    with open(os.path.join(source, "model.safetensors.index.json"), encoding="utf-8") as f:
        source_map = json.load(f)["weight_map"]
    with open(os.path.join(out, "model.safetensors.index.json"), encoding="utf-8") as f:
        out_map = json.load(f)["weight_map"]
    if source_map != out_map:
        raise AssertionError("weight maps differ")
    differing, linked, rewritten, kernel = [], 0, 0, None
    for shard in sorted(set(source_map.values())):
        if same_link(os.path.join(source, shard), os.path.join(out, shard)):
            linked += 1
            continue
        rewritten += 1
        with safe_open(os.path.join(source, shard), "pt") as a, safe_open(os.path.join(out, shard), "pt") as b:
            if set(a.keys()) != set(b.keys()):
                raise AssertionError(f"tensor names differ in {shard}")
            for name in a.keys():
                x, y = a.get_tensor(name), b.get_tensor(name)
                if name == PATCH_KERNEL:
                    kernel = dict(source_shape=list(x.shape), shape=list(y.shape), dtype=str(y.dtype))
                elif not (x.dtype == y.dtype and torch.equal(x, y)):
                    differing.append(name)
    config_changes = {}
    for name in sorted(os.listdir(source)):
        if name.endswith(".safetensors") or name in ("model.safetensors.index.json", fp.MARKER):
            continue
        if name in REWRITTEN:
            with open(os.path.join(source, name), encoding="utf-8") as f:
                a = flatten(json.load(f))
            with open(os.path.join(out, name), encoding="utf-8") as f:
                b = flatten(json.load(f))
            config_changes[name] = {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
        elif not same_link(os.path.join(source, name), os.path.join(out, name)):
            differing.append(name)
    return dict(shards_linked=linked, shards_rewritten=rewritten, patch_kernel=kernel, config_changes=config_changes,
                not_equal=differing, result="ok" if not differing and kernel else "failed")
