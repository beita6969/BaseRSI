import os
import time

import numpy as np
import torch



def record_name(row_id, sample):
    return f"{row_id.replace('/', '__')}__s{sample}.npz"


def shard_items(items, shard, n_shards):
    return list(items)[shard::n_shards]


def save_npz_atomic(path, arrays):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)
    return path


def capture_record(lab, row_id, sample, index, sampled_positions, seed):
    gen, think_len = lab.gen_ids(row_id, sample)
    ctx_base = lab.prep(row_id, "base", gen)
    ctx_text = lab.prep(row_id, "proc", gen)
    _, inserted = lab.align(ctx_base, ctx_text)
    prompt_len, total_len = ctx_base.P, ctx_base.T
    device, n_layers, hidden, tail = lab.device, lab.n_layers, lab.hidden, lab.tail_positions
    stream = torch.arange(prompt_len - tail, total_len, device=device)
    generator = torch.Generator().manual_seed(seed + 7 * index)
    n_sampled = min(sampled_positions, think_len)
    picked = torch.cat([torch.arange(tail), tail + torch.randperm(think_len, generator=generator)[:n_sampled].sort().values]).to(device)
    unit_input = np.zeros((n_layers, len(picked), hidden), np.float16)
    gap = np.zeros((n_layers, len(picked), hidden), np.float16)
    h_base, h_text = ctx_base.h0, ctx_text.h0
    state = lab.state
    for l in range(n_layers):
        h_hybrid = torch.cat([h_base[:, : prompt_len - tail], h_text[:, prompt_len - tail + inserted:]], 1)
        state["cap"] = dict(mlpin=lambda layer, o: o[0][stream][picked])
        state["out"] = {}
        h_hybrid_out = lab.run(ctx_base, start=l, stop=l + 1, h=h_hybrid)
        captured = state["out"]
        state["cap"] = {}
        state["out"] = {}
        h_base = lab.run(ctx_base, start=l, stop=l + 1, h=h_base)
        h_text = lab.run(ctx_text, start=l, stop=l + 1, h=h_text)
        block_gap = h_text[0][stream + inserted].float() - h_hybrid_out[0][stream].float()
        unit_input[l] = captured[("mlpin", l)].float().cpu().numpy()
        gap[l] = block_gap[picked].cpu().numpy()
        del captured, block_gap, h_hybrid, h_hybrid_out
    meta = dict(row=row_id, sample=sample, bench=lab.traces[row_id]["bench"], te=think_len, P=prompt_len, T=total_len, S=inserted)
    return meta, dict(U=unit_input, J=gap, samp=picked.cpu().numpy(), te=think_len)


def capture_gaps(lab, items, out_dir, sampled_positions, seed, shard=0, n_shards=1):
    os.makedirs(os.path.join(out_dir, "samp"), exist_ok=True)
    work = shard_items(items, shard, n_shards)
    for index, (row_id, sample) in enumerate(work):
        path = os.path.join(out_dir, record_name(row_id, sample))
        if os.path.exists(path):
            continue
        start = time.time()
        meta, samples = capture_record(lab, row_id, sample, index, sampled_positions, seed)
        save_npz_atomic(os.path.join(out_dir, "samp", os.path.basename(path)), samples)
        save_npz_atomic(path, meta)
        print(f"[{index + 1}/{len(work)}] {row_id} s{sample} think {meta['te']} {time.time() - start:.0f}s", flush=True)
        torch.cuda.empty_cache()
    return out_dir
