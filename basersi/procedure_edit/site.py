import glob
import os
import time

import numpy as np
import torch

from basersi.procedure_edit.gap_capture import record_name, save_npz_atomic, shard_items

POSITION_CHUNK = 256


def linear_blocks(lab):
    return [l for l in range(lab.n_layers) if lab.layers[l].block_type != "full_attention"]


def divergence(logp_from, logp_to):
    parts = [(logp_from[i: i + POSITION_CHUNK].exp() * (logp_from[i: i + POSITION_CHUNK] - logp_to[i: i + POSITION_CHUNK])).sum(-1)
             for i in range(0, logp_from.shape[0], POSITION_CHUNK)]
    return torch.cat(parts).cpu().numpy()


def patched_record(lab, row_id, sample):
    gen, think_len = lab.gen_ids(row_id, sample)
    ctx_base = lab.prep(row_id, "base", gen)
    ctx_text = lab.prep(row_id, "proc", gen)
    _, inserted = lab.align(ctx_base, ctx_text)
    device, state = lab.device, lab.state
    stream = torch.arange(ctx_base.P - lab.tail_positions, ctx_base.T, device=device)
    n_pred = min(think_len + 1, len(gen))
    positions = torch.arange(ctx_base.P - 1, ctx_base.P - 1 + n_pred, device=device)
    blocks = set(linear_blocks(lab))
    state["cap"] = dict(mix=lambda layer, o: o[0][stream + inserted].cpu() if layer in blocks else None)
    state["out"] = {}
    logp_teacher = lab.logp(lab.run(ctx_text)[0][positions + inserted])
    teacher_outputs = {l: state["out"][("mix", l)] for l in blocks}
    lab.clear()
    logp_student = lab.logp(lab.run(ctx_base)[0][positions])

    def put(cache):
        def patch(o):
            o = o.clone()
            o[0, stream] = cache.to(device).to(o.dtype)
            return o
        return patch
    state["patch"] = {("mix", l): put(teacher_outputs[l]) for l in blocks}
    try:
        logp_patched = lab.logp(lab.run(ctx_base)[0][positions])
    finally:
        lab.clear()
    return dict(row=row_id, sample=sample, bench=lab.traces[row_id]["bench"], te=think_len, P=ctx_base.P, T=ctx_base.T, S=inserted,
                kl_student=divergence(logp_teacher, logp_student), kl_patched=divergence(logp_teacher, logp_patched))


def measure(lab, items, out_dir, shard=0, n_shards=1):
    os.makedirs(out_dir, exist_ok=True)
    work = shard_items(items, shard, n_shards)
    for index, (row_id, sample) in enumerate(work):
        path = os.path.join(out_dir, record_name(row_id, sample))
        if os.path.exists(path):
            continue
        start = time.time()
        record = patched_record(lab, row_id, sample)
        save_npz_atomic(path, record)
        print(f"[{index + 1}/{len(work)}] {row_id} s{sample} think {record['te']} {time.time() - start:.0f}s", flush=True)
        torch.cuda.empty_cache()
    return out_dir


def load_records(directory):
    return [np.load(f) for f in sorted(glob.glob(os.path.join(directory, "*.npz")))]


def linear_fraction(records):
    values = [1 - float(r["kl_patched"].sum()) / float(r["kl_student"].sum()) for r in records if len(r["kl_student"]) and float(r["kl_student"].sum()) > 0]
    if not values:
        return dict(fraction=None, se=None, reads=0)
    return dict(fraction=float(np.mean(values)), se=float(np.std(values) / np.sqrt(len(values))), reads=len(values))
