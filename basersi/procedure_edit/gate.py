import numpy as np

KEY = "krt"


def block_count(arrays, prefix):
    n = 0
    while f"{prefix}_{n}" in arrays:
        n += 1
    return n


def add_key(arrays, slope, threshold, rank, unit_gate_scale):
    out = dict(arrays)
    n_blocks = block_count(out, "kr")
    slope, threshold = float(slope), float(threshold)
    for l in range(n_blocks):
        route = out[f"kr_{l}"].astype(np.float64).reshape(-1)
        constant = out[f"k_{l}"].astype(np.float64).reshape(-1)
        out[f"{KEY}_{l}"] = ((slope / unit_gate_scale) * (route - threshold * constant)).astype(np.float32)
    for l in range(n_blocks):
        assert f"up_FL_{l}_{rank}" not in out
        out[f"up_FL_{l}_{rank}"] = out[f"k_{l}"].astype(np.float32).reshape(-1)
        out[f"down_FL_{l}_{rank}"] = out[f"down_FJm_{l}"].astype(np.float32).reshape(-1, 1)
    return out
