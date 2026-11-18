import hashlib
import json
import os

import torch
from safetensors.torch import load_file, save_file

from basersi import fingerprint as fp
from basersi.view_edit.patch_kernel import PATCH_KERNEL, resample_patch_kernel

REWRITTEN = ("config.json", "preprocessor_config.json", "video_preprocessor_config.json")


def write_json(path, data, indent):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    os.replace(tmp, path)


def view_inputs(source, patch_size, min_tokens, max_tokens):
    return dict(model=fp.checkpoint_identity(source), patch_size=patch_size, min_image_tokens=min_tokens, max_image_tokens=max_tokens)


def kernel_shard(source):
    return fp.read_json(os.path.join(source, "model.safetensors.index.json"))["weight_map"][PATCH_KERNEL]


def link_unchanged(source, out, shard):
    linked = []
    for name in sorted(os.listdir(source)):
        if name in REWRITTEN or name == shard or name == fp.MARKER:
            continue
        os.symlink(os.path.realpath(os.path.join(source, name)), os.path.join(out, name))
        linked.append(name)
    return linked


def patched_config(source, patch_size):
    config = fp.read_json(os.path.join(source, "config.json"))
    config["vision_config"]["patch_size"] = patch_size
    return config


def image_processor_config(source, patch_size, merge, min_tokens, max_tokens):
    processor = fp.read_json(os.path.join(source, "preprocessor_config.json"))
    side = patch_size * merge
    processor["patch_size"] = patch_size
    processor["merge_size"] = merge
    processor["size"]["longest_edge"] = max_tokens * side ** 2
    processor["size"]["shortest_edge"] = min_tokens * side ** 2 if min_tokens else min(processor["size"]["shortest_edge"], processor["size"]["longest_edge"])
    return processor


def video_processor_config(source, patch_size, merge):
    path = os.path.join(source, "video_preprocessor_config.json")
    if not os.path.exists(path):
        return None
    video = fp.read_json(path)
    old_side = video["patch_size"] * video["merge_size"]
    video["patch_size"], video["merge_size"] = patch_size, merge
    video["size"]["longest_edge"] = int(video["size"]["longest_edge"] * (patch_size * merge) ** 2 / old_side ** 2)
    video["size"]["shortest_edge"] = min(video["size"]["shortest_edge"], video["size"]["longest_edge"])
    return video


def written_files(folder):
    return {name: fp.file_sha256(os.path.join(folder, name)) for name in sorted(os.listdir(folder))
            if name != fp.MARKER and not os.path.islink(os.path.join(folder, name)) and os.path.isfile(os.path.join(folder, name))}


def build_view_checkpoint(source, out, patch_size, min_tokens, max_tokens):
    from basersi.view_edit.check import check_view_checkpoint
    source, final = os.path.abspath(source), os.path.abspath(out).rstrip("/")
    if os.path.lexists(final):
        raise fp.OutputExists(f"{final} exists; the View Edit checkpoint is written to a new folder: choose another output path, "
                              f"or move it away: {fp.move_commands([final])}")
    inputs = view_inputs(source, patch_size, min_tokens, max_tokens)
    fp.require_stock(source, "the View Edit")
    edits = fp.marker_edits(source)
    out = final + ".partial"
    set_aside = fp.set_aside(out) if os.path.lexists(out) else None
    os.makedirs(out)
    shard = kernel_shard(source)
    linked = link_unchanged(source, out, shard)
    config = patched_config(source, patch_size)
    merge = config["vision_config"]["spatial_merge_size"]
    tensors = load_file(os.path.join(source, shard))
    old_patch = tensors[PATCH_KERNEL].shape[-1]
    tensors[PATCH_KERNEL], response_change = resample_patch_kernel(tensors[PATCH_KERNEL], patch_size)
    kernel_sha = hashlib.sha256(tensors[PATCH_KERNEL].contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    save_file(tensors, os.path.join(out, shard + ".tmp"), metadata={"format": "pt"})
    os.replace(os.path.join(out, shard + ".tmp"), os.path.join(out, shard))
    write_json(os.path.join(out, "config.json"), config, 2)
    processor = image_processor_config(source, patch_size, merge, min_tokens, max_tokens)
    write_json(os.path.join(out, "preprocessor_config.json"), processor, 4)
    video = video_processor_config(source, patch_size, merge)
    if video is not None:
        write_json(os.path.join(out, "video_preprocessor_config.json"), video, 4)
    check = check_view_checkpoint(source, out)
    if check["result"] != "ok":
        raise AssertionError(f"View Edit check failed on {out} ({check['not_equal']}); the folder is kept for inspection and the next build moves it aside")
    report = dict(checkpoint=final, source=source, kernel_shard=shard, old_patch_size=old_patch, patch_size=patch_size, merge_size=merge,
                  min_image_tokens=min_tokens, max_image_tokens=max_tokens, flat_patch_response_change=response_change,
                  image_pixels=dict(shortest_edge=processor["size"]["shortest_edge"], longest_edge=processor["size"]["longest_edge"]),
                  linked=len(linked), patch_kernel_sha256=kernel_sha, check=check,
                  written_sha256=written_files(out), inputs=fp.with_code(inputs, reads_rows=False), inputs_fingerprint=fp.fingerprint(inputs, reads_rows=False))
    report["marker"] = dict(file=fp.MARKER, content_sha256=fp.write_marker(out, edits + [dict(stage="view_edit", inputs_fingerprint=report["inputs_fingerprint"], report=report)]))
    os.rename(out, final)
    report["set_aside"] = set_aside
    return report
