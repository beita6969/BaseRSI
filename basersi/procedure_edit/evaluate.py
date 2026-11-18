import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from basersi.procedure_edit.gap_capture import record_name, save_npz_atomic, shard_items

POSITION_CHUNK = 256


def parse_replay_cell(text):
    name, spec = text.split("=", 1)
    fields = spec.split(":")
    lo, hi = (int(x) for x in fields[0].split("-"))
    rank, step = int(fields[1]), float(fields[2])
    options = dict(o.split("=") if "=" in o else (o, "1") for o in (fields[3].split(",") if len(fields) > 3 else []))
    unknown = sorted(set(options) - {"gate", "nobias"})
    if unknown or len(fields) > 4:
        raise ValueError(f"replay cell {text!r}: the form is name=lo-hi:units:step[:gate=G[,nobias]]; unknown {unknown or fields[4:]}")
    return name, dict(lo=lo, hi=hi, rank=rank, step=step, gate=options.get("gate", "ks"), bias="nobias" not in options)


def build_replay_cell(lab, arrays, cell):
    as_tensor = lambda z: torch.tensor(np.asarray(z, np.float32).reshape(-1), device=lab.device)
    blocks = {}
    for l in range(cell["lo"], cell["hi"] + 1):
        blocks[l] = dict(gate=as_tensor(arrays[f"{cell['gate']}_{l}"]), constant=as_tensor(arrays[f"k_{l}"]),
                         up=torch.stack([as_tensor(arrays[f"up_FL_{l}_{i}"]) for i in range(cell["rank"])]),
                         down=torch.stack([as_tensor(arrays[f"down_FL_{l}_{i}"]) for i in range(cell["rank"])]),
                         bias=as_tensor(arrays[f"down_FJm_{l}"]) if cell["bias"] else None)
    return blocks


def run_with_cell(lab, ctx, blocks, step, unit_gate_scale):
    store = {}
    state = lab.state

    def capture(layer, x):
        block = blocks.get(layer)
        if block is None:
            return None
        xx = x[0].float()
        gate_value = xx @ block["gate"]
        activations = xx @ block["up"].T
        out = (Fn.silu(unit_gate_scale * gate_value)[:, None] * activations) @ block["down"]
        if block["bias"] is not None:
            out = out + (Fn.silu(unit_gate_scale * (xx @ block["constant"])) * gate_value)[:, None] * block["bias"][None]
        store[layer] = (step * out).to(x.dtype)
        return None
    state["cap"] = dict(mlpin=capture)
    state["out"] = {}
    state["patch"] = {("mlp", l): (lambda o, l=l: o + store.pop(l)[None]) for l in blocks}
    try:
        return lab.run(ctx)
    finally:
        lab.clear()


def replay(lab, arrays, cell_texts, items, out_dir, unit_gate_scale, shard=0, n_shards=1):
    os.makedirs(out_dir, exist_ok=True)
    cells = {}
    for text in cell_texts:
        name, cell = parse_replay_cell(text)
        cells[name] = (build_replay_cell(lab, arrays, cell), cell["step"])
    device = lab.device
    work = shard_items(items, shard, n_shards)
    for index, (row_id, sample) in enumerate(work):
        path = os.path.join(out_dir, record_name(row_id, sample))
        if os.path.exists(path):
            continue
        start = time.time()
        gen, think_len = lab.gen_ids(row_id, sample)
        ctx_base = lab.prep(row_id, "base", gen)
        ctx_text = lab.prep(row_id, "proc", gen)
        _, inserted = lab.align(ctx_base, ctx_text)
        n_pred = min(think_len + 1, len(gen))
        positions = torch.arange(ctx_base.P - 1, ctx_base.P - 1 + n_pred, device=device)
        logp_base = lab.logp(lab.run(ctx_base)[0][positions])
        logp_text = lab.logp(lab.run(ctx_text)[0][positions + inserted])

        def chunked(fn, *tensors):
            return torch.cat([fn(*[t[i: i + POSITION_CHUNK] for t in tensors]) for i in range(0, tensors[0].shape[0], POSITION_CHUNK)]).cpu().numpy()
        kl = lambda a, b: chunked(lambda x, y: (x.exp() * (x - y)).sum(-1), a, b)
        record = dict(row=row_id, sample=sample, bench=lab.traces[row_id]["bench"], te=think_len)
        record["m/kl_xc"] = kl(logp_text, logp_base)
        names = ["m"]
        for name, (blocks, step) in cells.items():
            logp_cell = lab.logp(run_with_cell(lab, ctx_base, blocks, step, unit_gate_scale)[0][positions])
            record[f"{name}/kl_xc"] = kl(logp_text, logp_cell)
            names.append(name)
            del logp_cell
        record["names"] = np.array(names)
        save_npz_atomic(path, record)
        print(f"[{index + 1}/{len(work)}] {row_id} s{sample} think {think_len} {time.time() - start:.0f}s", flush=True)
        del logp_base, logp_text
        torch.cuda.empty_cache()
    return out_dir


def load_replay(directory):
    return [np.load(f) for f in sorted(glob.glob(os.path.join(directory, "*.npz")))]


def restoration_all(records, name):
    values = []
    for r in records:
        base = r["m/kl_xc"]
        if len(base) and base.sum() > 0:
            values.append(1 - r[f"{name}/kl_xc"].sum() / base.sum())
    return float(np.mean(values)) if values else float("nan")


def divergence_all(records, name):
    values = [float(r[f"{name}/kl_xc"].mean()) for r in records if len(r[f"{name}/kl_xc"])]
    return float(np.mean(values)) if values else float("nan")
