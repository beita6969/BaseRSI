import math

from basersi.procedure_edit.evaluate import divergence_all, restoration_all
from basersi.procedure_edit.gate import KEY


def candidate_name(step):
    return f"x{step:g}"


def replay_cell_text(step, rank, last_block):
    return f"{candidate_name(step)}=0-{last_block}:{rank + 1}:{step:g}:gate={KEY},nobias"


def spec_cell_text(step, rank, last_block):
    return f"0-{last_block}:{rank + 1}:{step:g}:{KEY}:nobias"


def finite(value):
    return value if math.isfinite(value) else None


def measure_steps(steps, records):
    return [dict(name=candidate_name(step), step=step, divergence=finite(divergence_all(records, candidate_name(step))),
                 restoration=finite(restoration_all(records, candidate_name(step)))) for step in steps]


def line_search(table):
    measured = [row for row in table if row["divergence"] is not None]
    if not measured:
        return None
    return min(measured, key=lambda row: (row["divergence"], row["step"]))
