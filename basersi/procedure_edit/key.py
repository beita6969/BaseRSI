import glob
import os
import time

import numpy as np
import torch

from basersi.procedure_edit.gap_capture import record_name, save_npz_atomic, shard_items
from basersi.procedure_edit.lab import in_fold

CLASS_IMAGE, CLASS_TEXT, CLASS_TAIL, CLASS_THINKING, CLASS_ANSWER = 1, 2, 3, 4, 5


def capture_unit_inputs(lab, items, out_dir, positions, seed, shard=0, n_shards=1):
    os.makedirs(out_dir, exist_ok=True)
    stop_block = lab.n_layers
    work = shard_items(items, shard, n_shards)
    device, n_layers, hidden, state = lab.device, lab.n_layers, lab.hidden, lab.state
    for index, (row_id, sample) in enumerate(work):
        path = os.path.join(out_dir, record_name(row_id, sample))
        if os.path.exists(path):
            continue
        start = time.time()
        gen, think_len = lab.gen_ids(row_id, sample)
        ctx = lab.prep(row_id, "base", gen)
        prompt_len, total_len = ctx.P, ctx.T
        by_class = lab.classes(ctx)
        generator = torch.Generator().manual_seed(seed + index)
        pick = lambda idx, n: idx[torch.randperm(len(idx), generator=generator)[:n].sort().values.to(idx.device)]
        thinking = torch.arange(prompt_len, prompt_len + think_len, device=device)
        answer = torch.arange(prompt_len + think_len, total_len, device=device)
        chosen = torch.cat([pick(by_class["img"], positions["image"]), pick(by_class["txt"], positions["text"]), by_class["tail"],
                            pick(thinking, positions["thinking"]), pick(answer, positions["answer"])])
        cls = torch.cat([torch.full((min(positions["image"], len(by_class["img"])),), CLASS_IMAGE), torch.full((min(positions["text"], len(by_class["txt"])),), CLASS_TEXT),
                         torch.full((lab.tail_positions,), CLASS_TAIL), torch.full((min(positions["thinking"], think_len),), CLASS_THINKING),
                         torch.full((min(positions["answer"], len(answer)),), CLASS_ANSWER)])
        thinking_index = torch.where((chosen >= prompt_len) & (chosen < prompt_len + think_len), chosen - prompt_len, torch.full_like(chosen, -1))
        state["cap"] = dict(mlpin=lambda layer, o: o[0][chosen].float().cpu())
        state["out"] = {}
        lab.run(ctx, stop=stop_block)
        X = np.zeros((n_layers, len(chosen), hidden), np.float16)
        X[:stop_block] = torch.stack([state["out"][("mlpin", l)] for l in range(stop_block)]).numpy().astype(np.float16)
        lab.clear()
        save_npz_atomic(path, dict(X=X, cls=cls.numpy(), tix=thinking_index.cpu().numpy(), row=row_id, sample=sample))
        print(f"[{index + 1}/{len(work)}] {row_id} s{sample} T {total_len} positions {len(chosen)} {time.time() - start:.0f}s", flush=True)
    return out_dir


def load_unit_inputs(directory):
    return [{k: v for k, v in np.load(f).items()} for f in sorted(glob.glob(os.path.join(directory, "*.npz")))]


def weighted_ridge(X, weights, target, ridge, device):
    gram = (X * weights[:, None]).T @ X
    return torch.linalg.solve(gram + ridge * gram.diagonal().mean() * torch.eye(X.shape[1], device=device), (X * weights[:, None]).T @ target)


def fit_position_keys(records, base_arrays, ridge, val_fold, n_folds, device):
    n_blocks = records[0]["X"].shape[0]
    is_val = np.array([in_fold(str(r["row"]), val_fold, n_folds) for r in records])
    cls = np.concatenate([r["cls"] for r in records])
    record_of = np.concatenate([np.full(len(r["cls"]), i) for i, r in enumerate(records)])
    arrays = dict(base_arrays)
    val = torch.tensor(is_val[record_of], device=device)
    c = torch.tensor(cls, device=device)
    train = ~val
    stream = (c >= CLASS_TAIL).float()
    stream_weights = torch.where(stream > 0, 0.5 / stream[train].sum(), 0.5 / (1 - stream[train]).sum())
    class_weights = torch.zeros_like(stream)
    for q in range(CLASS_IMAGE, CLASS_ANSWER + 1):
        class_weights = torch.where(c == q, 0.2 / max(1, int((c[train] == q).sum())), class_weights)
    for l in range(n_blocks):
        X = torch.tensor(np.concatenate([r["X"][l].astype(np.float32) for r in records], 0), device=device)
        stream_key = weighted_ridge(X[train], stream_weights[train], stream[train], ridge, device)
        constant_key = weighted_ridge(X[train], class_weights[train], torch.ones_like(stream[train]), ridge, device)
        arrays[f"ks_{l}"] = stream_key.cpu().numpy().astype(np.float32)
        arrays[f"k_{l}"] = constant_key.cpu().numpy().astype(np.float32)
        del X
    return arrays


def fit_route_key(routed_records, unrouted_records, fit_arrays, ridge, val_fold, n_folds, device):
    records = [dict(r, routed=1) for r in routed_records] + [dict(r, routed=0) for r in unrouted_records]
    n_blocks = records[0]["X"].shape[0]
    cls = np.concatenate([r["cls"] for r in records])
    label = np.concatenate([np.full(len(r["cls"]), r["routed"]) for r in records])
    record_of = np.concatenate([np.full(len(r["cls"]), i) for i, r in enumerate(records)])
    held = np.array([in_fold(str(r["row"]), val_fold, n_folds) for r in records])[record_of]
    stream = cls >= CLASS_TAIL
    group = np.where(~stream, 2, np.where(label == 1, 0, 1))
    target = (group == 0).astype(np.float32)
    train = ~held
    weights = np.zeros(len(target), np.float32)
    for g in range(3):
        m = train & (group == g)
        weights[m] = 1.0 / 3 / max(1, m.sum())
    arrays = dict(fit_arrays)
    target_t = torch.tensor(target, device=device)
    weights_t = torch.tensor(weights, device=device)
    train_t = torch.tensor(train, device=device)
    for l in range(n_blocks):
        X = torch.tensor(np.concatenate([r["X"][l].astype(np.float32) for r in records]), device=device)
        X_train, w_train, y_train = X[train_t], weights_t[train_t], target_t[train_t]
        x_mean = (X_train * w_train[:, None]).sum(0)
        y_mean = (y_train * w_train).sum()
        Xc = X_train - x_mean
        gram = (Xc * w_train[:, None]).T @ Xc
        w = torch.linalg.solve(gram + ridge * gram.diagonal().mean() * torch.eye(gram.shape[0], device=device), (Xc * w_train[:, None]).T @ (y_train - y_mean))
        intercept = float(y_mean - x_mean @ w)
        arrays[f"kr_{l}"] = (w.cpu().numpy() + intercept * arrays[f"k_{l}"].reshape(-1)).astype(np.float32)
        del X, X_train, gram
    counts = dict(routed=len(routed_records), unrouted=len(unrouted_records),
                  training_positions=dict(routed_stream=int((train & (group == 0)).sum()), unrouted_stream=int((train & (group == 1)).sum()), context=int((train & (group == 2)).sum())))
    return arrays, counts
