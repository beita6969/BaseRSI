import filecmp
import hashlib
import json
import os
import shutil
import struct

import numpy as np
import torch

from basersi import fingerprint as fp

PARAM_PREFIX = "model.language_model."
BYTE_CHUNK = 1 << 28


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


class CheckpointFolder:
    def __init__(self, directory, write=False):
        self.directory, self.write = directory, write
        with open(os.path.join(directory, "model.safetensors.index.json"), encoding="utf-8") as f:
            self.weight_map = json.load(f)["weight_map"]
        self.headers, self.maps = {}, {}

    def tensor(self, name):
        shard = self.weight_map[name]
        if shard not in self.headers:
            header, offset = safetensors_header(os.path.join(self.directory, shard))
            self.headers[shard] = (header, offset)
            self.maps[shard] = np.memmap(os.path.join(self.directory, shard), dtype=np.uint8, mode="r+" if self.write else "r")
        header, offset = self.headers[shard]
        entry = header[name]
        begin, end = entry["data_offsets"]
        assert entry["dtype"] == "BF16", (name, entry["dtype"])
        raw = self.maps[shard][offset + begin: offset + end].view(np.uint16).reshape(entry["shape"])
        return torch.from_numpy(raw).view(torch.bfloat16) if self.write else torch.from_numpy(np.array(raw)).view(torch.bfloat16)

    def flush(self):
        if self.write:
            for m in self.maps.values():
                m.flush()


def param_name(param):
    return PARAM_PREFIX + param + ".weight" if not param.startswith(PARAM_PREFIX) else param


def op_neurons(op):
    neurons = op.get("sel", op).get("neurons")
    return [neurons] if isinstance(neurons, int) else list(neurons)


def touched_rows(spec):
    touched = {}
    for op in spec:
        touched.setdefault(param_name(op["param"]), set()).update(op_neurons(op))
    return {k: sorted(v) for k, v in touched.items()}


def array_file(name, spec_path, remap=None):
    if name in (remap or {}):
        return remap[name]
    return name if os.path.isabs(name) else os.path.join(os.path.dirname(os.path.abspath(spec_path)), name)


def spec_array_files(spec):
    return sorted({op[k]["npz"] for op in spec for k in ("value", "alpha") if isinstance(op.get(k), dict) and "npz" in op[k]})


def spec_inputs(base, spec_path, remap=None):
    spec = load_spec(spec_path)
    missing = [(name, array_file(name, spec_path, remap)) for name in spec_array_files(spec) if not os.path.isfile(array_file(name, spec_path, remap))]
    if missing:
        raise fp.config.MissingInput(f"the spec {spec_path} names array files that do not exist: "
                                     + "; ".join(f"{name} -> {path}" for name, path in missing[:3])
                                     + f"{f' and {len(missing) - 3} more' if len(missing) > 3 else ''}; keep them next to the spec or give --remap")
    return dict(base=fp.checkpoint_identity(base), spec=fp.file_sha256(spec_path),
                arrays={name: fp.file_sha256(array_file(name, spec_path, remap)) for name in spec_array_files(spec)})


def load_spec(spec_path, arg="the spec"):
    return fp.given_json(spec_path, arg, fp.SPEC_EXPECTED, fp.is_spec)


class ArrayStore:
    def __init__(self, spec_path, remap=None):
        self.cache, self.remap, self.spec_path = {}, remap or {}, spec_path

    def value(self, v):
        if isinstance(v, dict) and "npz" in v:
            path = array_file(v["npz"], self.spec_path, self.remap)
            z = self.cache.setdefault(path, np.load(path))
            return torch.tensor(np.asarray(z[v["key"]], dtype=np.float32))
        return torch.tensor(np.asarray(v, dtype=np.float32))


def rows_get(W, leaf, index):
    return W[index].clone() if leaf != "down_proj" else W[:, index].T.clone()


def rows_set(W, leaf, index, new):
    if leaf == "down_proj":
        W[:, index] = new.T
    else:
        W[index] = new


@torch.no_grad()
def apply_spec(folder, spec, store):
    for op in spec:
        name = param_name(op["param"])
        leaf = name.split(".")[-2]
        W = folder.tensor(name)
        neurons = op_neurons(op)
        index = torch.tensor(neurons)
        x = rows_get(W, leaf, index).float()
        if op["op"] == "set":
            new = store.value(op["value"]).reshape(len(neurons), -1)
        elif op["op"] == "scale":
            alpha = op["alpha"]
            new = x * (float(alpha) if not isinstance(alpha, (list, dict)) else store.value(alpha).reshape(len(neurons), 1))
        elif op["op"] == "add":
            new = x + store.value(op["value"]).reshape(len(neurons), -1)
        else:
            raise fp.InputRefused(f"the spec holds an unknown op {op['op']!r}")
        rows_set(W, leaf, index, new.to(torch.bfloat16))


@torch.no_grad()
def restore_rows(folder, reference, touched):
    for name, neurons in touched.items():
        leaf = name.split(".")[-2]
        index = torch.tensor(neurons)
        rows_set(folder.tensor(name), leaf, index, rows_get(reference.tensor(name), leaf, index))


def byte_check(directory, reference, touched):
    n_tensors = n_bytes = n_changed = 0
    with open(os.path.join(reference, "model.safetensors.index.json"), encoding="utf-8") as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    for shard in shards:
        header_s, offset_s = safetensors_header(os.path.join(directory, shard))
        header_r, offset_r = safetensors_header(os.path.join(reference, shard))
        assert header_s == header_r and offset_s == offset_r, ("header differs", shard)
        map_s = np.memmap(os.path.join(directory, shard), dtype=np.uint8, mode="r")
        map_r = np.memmap(os.path.join(reference, shard), dtype=np.uint8, mode="r")
        assert map_s.shape == map_r.shape, shard
        assert np.array_equal(map_s[:offset_s], map_r[:offset_r]), ("header bytes", shard)
        for name, entry in header_s.items():
            begin, end = entry["data_offsets"]
            n_tensors += 1
            if name not in touched:
                for s in range(offset_s + begin, offset_s + end, BYTE_CHUNK):
                    assert np.array_equal(map_s[s:min(s + BYTE_CHUNK, offset_s + end)], map_r[s:min(s + BYTE_CHUNK, offset_s + end)]), ("bytes differ outside the spec", name)
                n_bytes += end - begin
                continue
            S = map_s[offset_s + begin: offset_s + end].view(np.uint16).reshape(entry["shape"])
            Rr = map_r[offset_r + begin: offset_r + end].view(np.uint16).reshape(entry["shape"])
            leaf = name.split(".")[-2]
            mask = np.ones(entry["shape"][0] if leaf != "down_proj" else entry["shape"][1], bool)
            mask[touched[name]] = False
            if leaf == "down_proj":
                assert np.array_equal(S[:, mask], Rr[:, mask]), ("columns differ outside the spec", name)
                n_changed += int((S[:, ~mask] != Rr[:, ~mask]).any(0).sum())
            else:
                assert np.array_equal(S[mask], Rr[mask]), ("rows differ outside the spec", name)
                n_changed += int((S[~mask] != Rr[~mask]).any(1).sum())
            n_bytes += end - begin
    for f in sorted(os.listdir(reference)):
        if f.endswith(".safetensors") or f == fp.MARKER or os.path.isdir(os.path.join(reference, f)):
            continue
        with open(os.path.join(directory, f), "rb") as a, open(os.path.join(reference, f), "rb") as b:
            assert a.read() == b.read(), ("file differs", f)
    return n_tensors, n_bytes, n_changed


def sha_of_rows(folder, touched):
    h = hashlib.sha256()
    for name in sorted(touched):
        leaf = name.split(".")[-2]
        x = rows_get(folder.tensor(name), leaf, torch.tensor(touched[name]))
        h.update(name.encode())
        h.update(x.view(torch.int16).numpy().tobytes())
    return h.hexdigest()


def copy_resolved(source, target):
    assert not os.path.lexists(target), target
    edits = fp.marker_edits(source)
    top = os.path.abspath(source)
    os.makedirs(target)
    fp.write_marker(target, edits, pending=dict(stage="copy", source=os.path.realpath(source)))
    shutil.copytree(source, target, symlinks=False, dirs_exist_ok=True, copy_function=shutil.copy2,
                    ignore=lambda d, names: [fp.MARKER] if os.path.abspath(d) == top else [])
    n = 0
    for root, _, files in os.walk(source, followlinks=True):
        for f in files:
            if root == source and f == fp.MARKER:
                continue
            src = os.path.join(root, f)
            dst = os.path.join(target, os.path.relpath(src, source))
            assert filecmp.cmp(src, dst, shallow=False), ("copy differs", src)
            n += 1
    fp.write_marker(target, edits)
    return n


SPEC_STAGES = ("export", "student", "patch")


def check_record(directory, reference, spec_path, previous_spec_path):
    base = fp.edit_keys(fp.marker_edits(reference))
    marker = fp.read_marker(directory)
    fresh = f"or patch a fresh copy (basersi export checkpoint --base {reference} --out NEW_FOLDER)"
    if marker is None:
        if base:
            raise fp.InputRefused(f"{directory} carries no {fp.MARKER} while {reference} does: it is not a copy of that reference; {fresh}")
        held = []
    else:
        held = fp.edit_keys(fp.marker_edits(directory))
    if held == base:
        if previous_spec_path:
            raise fp.InputRefused(f"{directory} holds no edit beyond {reference} ({fp.MARKER}), so there is no previous spec to restore: drop --previous-spec")
        return
    if held[:-1] != base:
        raise fp.InputRefused(f"the edits recorded in {directory}/{fp.MARKER} do not continue those of {reference}: it is not a copy of that reference; {fresh}")
    last_edit = fp.read_marker(directory)["edits"][-1]
    last = last_edit["report"]
    want = (last.get("inputs") or {}).get("spec")
    if last_edit.get("stage") not in SPEC_STAGES or not want:
        source = last.get("checkpoint") or last.get("folder")
        raise fp.InputRefused(f"{directory} holds a {last_edit.get('stage')} edit beyond {reference} ({fp.MARKER}), not a spec: it is a copy of "
                         f"{source or 'another checkpoint'}, not of {reference}; give that folder as --reference, {fresh}")
    if previous_spec_path and not os.path.isfile(previous_spec_path):
        raise fp.InputRefused(f"--previous-spec {previous_spec_path} does not exist; give the spec recorded in {directory}/{fp.MARKER} ({last.get('spec')}), {fresh}")
    if not previous_spec_path or fp.file_sha256(previous_spec_path) != want:
        given = f"--previous-spec {previous_spec_path} has sha256 {fp.file_sha256(previous_spec_path)[:16]}" if previous_spec_path else "no --previous-spec was given"
        raise fp.InputRefused(f"{directory} already holds the spec {last.get('spec')} (sha256 {want[:16]}, {fp.MARKER}) and {given}; writing {spec_path} over it "
                         f"without restoring that spec's rows would mix the two: give --previous-spec with the recorded spec, {fresh}")


def check_separate_copy(directory, reference):
    with open(os.path.join(reference, "model.safetensors.index.json"), encoding="utf-8") as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    for shard in shards:
        mine, theirs = os.path.join(directory, shard), os.path.join(reference, shard)
        if os.path.islink(mine) or not os.path.isfile(mine):
            raise fp.InputRefused(f"{mine} is a link or not a regular file: the spec is written only into a full copy (export checkpoint makes one)")
        a, b = os.stat(mine), os.stat(theirs)
        if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
            raise fp.InputRefused(f"{mine} is the same file as {theirs}: the folder to edit must be a separate copy of the reference")


def patch_folder(directory, reference, spec_path, previous_spec_path=None, remap=None, stage="patch", extra=None):
    check_separate_copy(directory, reference)
    check_record(directory, reference, spec_path, previous_spec_path)
    spec = load_spec(spec_path)
    inputs = spec_inputs(reference, spec_path, remap)
    edits = fp.marker_edits(reference)
    fp.write_marker(directory, edits, pending=dict(stage=stage, spec_sha256=inputs["spec"]))
    folder, ref = CheckpointFolder(directory, write=True), CheckpointFolder(reference)
    if previous_spec_path:
        restore_rows(folder, ref, touched_rows(load_spec(previous_spec_path, "--previous-spec")))
        folder.flush()
    touched = touched_rows(spec)
    apply_spec(folder, spec, ArrayStore(spec_path, remap))
    folder.flush()
    n_tensors, n_bytes, n_changed = byte_check(directory, reference, touched)
    report = dict(folder=directory, reference=reference, spec=spec_path, spec_sha256=inputs["spec"], restored_from=previous_spec_path,
                  tensors_touched=len(touched), rows_touched=sum(len(v) for v in touched.values()), rows_changed=n_changed,
                  sha_changed=sha_of_rows(CheckpointFolder(directory), touched),
                  check=dict(tensors=n_tensors, bytes_equal_outside_spec=n_bytes, result="ok"))
    report.update(extra or {})
    report.update(inputs=fp.with_code(inputs, reads_rows=False), inputs_fingerprint=fp.fingerprint(inputs, reads_rows=False))
    report["marker"] = dict(file=fp.MARKER, content_sha256=fp.write_marker(directory, edits + [dict(stage=stage, inputs_fingerprint=report["inputs_fingerprint"], report=report)]))
    return report


def export_checkpoint(base, spec_path, out_dir, remap=None, extra=None, key=None):
    out_dir = os.path.abspath(out_dir).rstrip("/")
    if os.path.lexists(out_dir):
        raise fp.OutputExists(f"{out_dir} exists; the export is written to a new folder: {fp.way_forward([out_dir], key)}")
    partial = out_dir + ".partial"
    set_aside = fp.set_aside(partial) if os.path.lexists(partial) else None
    n_files = copy_resolved(base, partial)
    report = patch_folder(partial, base, spec_path, remap=remap, stage="export", extra=dict(extra or {}, folder=out_dir, files_copied=n_files))
    os.rename(partial, out_dir)
    report["set_aside"] = set_aside
    return report
