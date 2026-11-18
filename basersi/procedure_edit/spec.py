import json
import os

from basersi import fingerprint as fp


CELL_OPTIONS = ("nobias",)


def parse_cell(text):
    fields = text.split(":")
    lo, hi = (int(x) for x in fields[0].split("-"))
    rank, step = int(fields[1]), float(fields[2])
    gate = fields[3] if len(fields) > 3 else "ks"
    options = fields[4].split(",") if len(fields) > 4 else []
    unknown = [o for o in options if o not in CELL_OPTIONS]
    if unknown or len(fields) > 5:
        raise fp.InputRefused(f"cell {text!r}: the form is lo-hi:units:step[:gate[:nobias]]; unknown {unknown or fields[5:]}")
    return dict(lo=lo, hi=hi, rank=rank, step=step, gate=gate, bias="nobias" not in options)


def unit_ops(block, neuron, gate_key, up_key, down_key, step, arrays_ref, keys, unit_gate_scale):
    prefix = f"layers.{block}.mlp."
    assert gate_key in keys and up_key in keys and down_key in keys, (gate_key, up_key, down_key)
    value = lambda k: {"npz": arrays_ref, "key": k}
    return [{"op": "set", "param": prefix + "gate_proj", "sel": {"neurons": [neuron]}, "value": value(gate_key)},
            {"op": "scale", "param": prefix + "gate_proj", "sel": {"neurons": [neuron]}, "alpha": float(unit_gate_scale)},
            {"op": "set", "param": prefix + "up_proj", "sel": {"neurons": [neuron]}, "value": value(up_key)},
            {"op": "set", "param": prefix + "down_proj", "sel": {"neurons": [neuron]}, "value": value(down_key)},
            {"op": "scale", "param": prefix + "down_proj", "sel": {"neurons": [neuron]}, "alpha": float(step)}]


def cell_ops(cell_text, keys, free, arrays_ref, unit_gate_scale):
    cell = parse_cell(cell_text)
    ops, neurons = [], {}
    for l in range(cell["lo"], cell["hi"] + 1):
        neurons[str(l)] = []
        if cell["bias"]:
            ops += unit_ops(l, free[l][0], f"k_{l}", f"{cell['gate']}_{l}", f"down_FJm_{l}", cell["step"], arrays_ref, keys, unit_gate_scale)
            neurons[str(l)].append(free[l][0])
        for i in range(cell["rank"]):
            ops += unit_ops(l, free[l][1 + i], f"{cell['gate']}_{l}", f"up_FL_{l}_{i}", f"down_FL_{l}_{i}", cell["step"], arrays_ref, keys, unit_gate_scale)
            neurons[str(l)].append(free[l][1 + i])
    return ops, neurons


def load_free(path):
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def write_spec(path, cell_text, keys, free, arrays_ref, unit_gate_scale):
    ops, neurons = cell_ops(cell_text, keys, free, os.path.relpath(os.path.abspath(arrays_ref), os.path.dirname(os.path.abspath(path))), unit_gate_scale)
    text = json.dumps(ops)
    assert not os.path.exists(path), path
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return path, neurons


def block_of(op):
    return int(op["param"].split(".")[1])


def key_block(key):
    return key.rsplit("_", 1)[0]


def describe(ops):
    if not ops or len(ops) % 5:
        raise fp.InputRefused("a Procedure Edit spec is a list of five-op units")
    blocks, steps, gates = {}, set(), set()
    for i in range(0, len(ops), 5):
        gate_op, _, up_op, down_op, step_op = ops[i: i + 5]
        l = block_of(gate_op)
        steps.add(float(step_op["alpha"]))
        is_bias = down_op["value"]["key"] == f"down_FJm_{l}"
        gates.add(key_block(up_op["value"]["key"] if is_bias else gate_op["value"]["key"]))
        blocks.setdefault(l, [0, 0])[0 if is_bias else 1] += 1
    regular = {v[1] for v in blocks.values()}
    with_bias = {v[0] for v in blocks.values()}
    if len(steps) != 1 or len(gates) != 1 or len(regular) != 1 or len(with_bias) != 1 or sorted(blocks) != list(range(min(blocks), max(blocks) + 1)):
        raise fp.InputRefused("the spec is not one uniform cell (same units, gate and step in every block of one range)")
    lo, hi, units, step, gate = min(blocks), max(blocks), regular.pop(), steps.pop(), gates.pop()
    text = f"{lo}-{hi}:{units}:{step:g}:{gate}" + ("" if with_bias.pop() else ":nobias")
    return dict(cell=text, blocks=[lo, hi], units=units, step=step, gate=gate)


def describe_file(path):
    with open(path, encoding="utf-8") as f:
        return describe(json.load(f))
