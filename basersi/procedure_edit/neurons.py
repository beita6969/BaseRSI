import json
import os

import numpy as np
import torch


def neuron_write_cost(lab, items):
    state, device = lab.state, lab.device
    activity = {l: torch.zeros(lab.layers[l].mlp.down_proj.weight.shape[1], dtype=torch.float64, device=device) for l in range(lab.n_layers)}
    positions = 0

    def capture(layer, x):
        activity[layer].add_(x[0].float().abs().sum(0).double())
        return None
    for row_id, sample in items:
        gen, _ = lab.gen_ids(row_id, sample)
        ctx = lab.prep(row_id, "base", gen)
        state["cap"] = dict(neu=capture)
        lab.run(ctx)
        lab.clear()
        positions += ctx.T
    cost = {}
    for l in range(lab.n_layers):
        mean_abs = (activity[l] / positions).cpu().numpy()
        column_norm = lab.layers[l].mlp.down_proj.weight.float().norm(dim=0).cpu().numpy()
        cost[l] = mean_abs * column_norm
    return cost, positions


def free_neurons(cost, used, per_block):
    out = {}
    for l, write in cost.items():
        order = np.argsort(write)
        out[l] = [int(x) for x in order if int(x) not in used.get(l, set())][:per_block]
    return out


def measurement_items(fit_items, unrouted_select_items, n_routed, n_unrouted, unrouted_stride):
    return list(fit_items)[:n_routed] + list(unrouted_select_items)[::unrouted_stride][:n_unrouted]


def load_used(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return {int(k): set(v) for k, v in json.load(f).items()}


def select_free_neurons(lab, fit_items, unrouted_select_items, used_path, out_path, cfg):
    assert not os.path.exists(out_path), out_path
    items = measurement_items(fit_items, unrouted_select_items, cfg["n_routed_reads"], cfg["n_unrouted_reads"], cfg["unrouted_stride"])
    cost, positions = neuron_write_cost(lab, items)
    free = free_neurons(cost, load_used(used_path), cfg["candidates_per_block"])
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(free, f)
    os.replace(tmp, out_path)
    return out_path
