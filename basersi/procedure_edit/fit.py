import glob
import os

import numpy as np
import torch

from basersi.procedure_edit.gap_capture import save_npz_atomic
from basersi.procedure_edit.lab import in_fold

SCORED_RANKS = (1, 2, 4, 8, 16, 32, 64)


def aggregate_records(capture_dirs_with_tags):
    named = []
    for directory, tag in capture_dirs_with_tags:
        for path in glob.glob(os.path.join(directory, "*.npz")):
            base = os.path.basename(path)[: -len(".npz")]
            named.append((f"{base}__{tag}.npz" if tag else f"{base}.npz", path, os.path.join(directory, "samp", base + ".npz")))
    return [(meta, samp) for _, meta, samp in sorted(named)]


def load_records(records):
    rows, samples = [], []
    for meta_path, samp_path in records:
        meta = np.load(meta_path)
        rows.append(str(meta["row"]))
        z = np.load(samp_path)
        samples.append(dict(U=z["U"], J=z["J"]))
    return rows, samples


def output_directions(fitted, rank, decomposition, oversampling, power_iterations):
    if decomposition == "exact":
        values, vectors = torch.linalg.eigh(fitted.double().T @ fitted.double())
        return vectors[:, torch.argsort(values, descending=True)[:rank]].to(fitted.dtype)
    assert decomposition == "randomized", decomposition
    return torch.svd_lowrank(fitted, q=rank + oversampling, niter=power_iterations)[2][:, :rank]


def ridge_reduced_rank_block(X, Y, is_val, is_tail, rank, lambdas, device, decomposition, oversampling, power_iterations):
    X_train, Y_train, X_val, Y_val = X[~is_val], Y[~is_val], X[is_val], Y[is_val]
    x_mean = X_train.mean(0)
    bias = Y_train[~is_tail[~is_val]].mean(0)
    x_dir = x_mean / x_mean.norm()
    features = lambda Z: (Z - x_mean) - ((Z - x_mean) @ x_dir)[:, None] * x_dir[None]
    Xc, Yc, Xvc, Yvc = features(X_train), Y_train - bias, features(X_val), Y_val - bias
    hidden = Xc.shape[1]
    gram = Xc.T @ Xc / len(Xc)
    cross = Xc.T @ Yc / len(Xc)
    total = float(Y_val.pow(2).sum())
    scored_rank = [r for r in SCORED_RANKS if r <= rank][-1]
    best = None
    for lam in lambdas:
        B = torch.linalg.solve(gram + lam * gram.diagonal().mean() * torch.eye(hidden, device=device), cross)
        V = output_directions(Xc @ B, rank, decomposition, oversampling, power_iterations)
        BV = B @ V
        explained = 1 - float((Yvc - (Xvc @ BV[:, :scored_rank]) @ V[:, :scored_rank].T).pow(2).sum()) / total
        if best is None or explained > best[0]:
            best = (explained, lam, BV.clone(), V.clone())
        del B
    explained, lam, BV, V = best
    scale = (Xc @ BV).std(0).clamp_min(1e-8)
    up = (BV / scale[None]).T.cpu().numpy()
    down = (V * scale[None]).T.cpu().numpy()
    return up, down, bias.cpu().numpy(), lam, explained


def fit_cell(records, base_arrays, rank, lambdas, val_fold, n_folds, device, seed, tail_positions, unit_gate_scale, oversampling, power_iterations,
             decomposition, n_threads, blocks=None):
    torch.set_num_threads(n_threads)
    settings = dict(rank=rank, lambdas=lambdas, val_fold=val_fold, n_folds=n_folds, device=device, tail_positions=tail_positions,
                    unit_gate_scale=unit_gate_scale, oversampling=oversampling, power_iterations=power_iterations, blocks=blocks, decomposition=decomposition)
    if seed is None:
        return fit_blocks(records, base_arrays, **settings)
    devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return fit_blocks(records, base_arrays, **settings)


def fit_blocks(records, base_arrays, rank, lambdas, val_fold, n_folds, device, tail_positions, unit_gate_scale, oversampling, power_iterations, blocks, decomposition):
    rows, samples = load_records(records)
    is_val_record = np.array([in_fold(r, val_fold, n_folds) for r in rows])
    n_blocks, _, hidden = samples[0]["U"].shape
    up_all = np.zeros((n_blocks, rank, hidden), np.float32)
    down_all = np.zeros((n_blocks, rank, hidden), np.float32)
    bias_all = np.zeros((n_blocks, hidden), np.float32)
    chosen = {}
    first, last = blocks if blocks is not None else (0, n_blocks - 1)
    for l in range(first, last + 1):
        X = torch.tensor(np.concatenate([s["U"][l].astype(np.float32) for s in samples]), device=device)
        Y = torch.tensor(np.concatenate([s["J"][l].astype(np.float32) for s in samples]), device=device)
        record_of = np.concatenate([np.full(s["U"].shape[1], i) for i, s in enumerate(samples)])
        tail = np.concatenate([np.arange(s["U"].shape[1]) < tail_positions for s in samples])
        is_val = torch.tensor(is_val_record[record_of], device=device)
        is_tail = torch.tensor(tail, device=device)
        up_all[l], down_all[l], bias_all[l], lam, explained = ridge_reduced_rank_block(X, Y, is_val, is_tail, rank, lambdas, device, decomposition, oversampling, power_iterations)
        chosen[l] = dict(ridge=lam, explained_val=explained, n_train=int((~is_val).sum()), n_val=int(is_val.sum()))
        print(f"{l:2d} ridge {lam} explained (val) {explained:+.3f}", flush=True)
        del X, Y
    arrays = dict(base_arrays)
    for l in range(first, last + 1):
        arrays[f"down_FJm_{l}"] = (bias_all[l] / unit_gate_scale).astype(np.float32)
        for i in range(rank):
            arrays[f"up_FL_{l}_{i}"] = up_all[l, i]
            arrays[f"down_FL_{l}_{i}"] = (down_all[l, i] / unit_gate_scale).astype(np.float32)
    arrays = {k: (np.asarray(v, np.float32).reshape(-1, 1) if k.startswith("down_") else np.asarray(v, np.float32).reshape(-1)) for k, v in arrays.items()}
    return arrays, dict(records=len(rows), val_records=int(is_val_record.sum()), blocks=chosen)


def load_arrays(path):
    return dict(np.load(path))


def array_keys(path):
    with np.load(path) as z:
        return set(z.files)


def save_arrays(path, arrays):
    assert not os.path.exists(path), path
    return save_npz_atomic(path, arrays)
