import numpy as np

from basersi import fingerprint as fp

from basersi.procedure_edit.lab import in_fold, pick_sample, window_samples


def signal_rows(traces, signal_roles):
    return sorted(r for r, v in traces.items() if isinstance(v, dict) and v.get("role") in signal_roles)


def is_routed(traces, row_id):
    return bool(traces[row_id].get("prefix"))


def one_read_each(traces, row_ids, window):
    out = []
    for r in row_ids:
        k = pick_sample(traces, r, **window)
        if k is not None:
            out.append((r, k))
    return out


def several_reads_each(traces, row_ids, n_reads, window):
    out = []
    for r in row_ids:
        samples = window_samples(traces, r, window["gen_lo"], window["gen_hi"], window["prompt_hi"])
        first = pick_sample(traces, r, **window)
        if first is None:
            continue
        ordered = [first] + [k for k in samples if k != first]
        out += [(r, k) for k in ordered[:n_reads]]
    return out


def permuted_per_benchmark(traces, row_ids, rng, window):
    out = {}
    for bench in sorted({traces[r]["bench"] for r in row_ids}):
        candidates = one_read_each(traces, [r for r in row_ids if traces[r]["bench"] == bench], window)
        out[bench] = (candidates, rng.permutation(len(candidates)))
    return out


def first_per_benchmark(permuted, n_per_benchmark, benchmarks):
    out = []
    for bench in sorted(permuted):
        if bench not in benchmarks:
            continue
        candidates, order = permuted[bench]
        out += [candidates[i] for i in sorted(order[:n_per_benchmark])]
    return out


def round_robin_continuation(continuations, benchmarks, n_needed):
    added, position = [], {b: 0 for b in benchmarks}
    while len(added) < n_needed and any(position[b] < len(continuations.get(b, [])) for b in benchmarks):
        for bench in benchmarks:
            if len(added) < n_needed and position[bench] < len(continuations.get(bench, [])):
                added.append(continuations[bench][position[bench]])
                position[bench] += 1
    return added


def compile_lists(traces, benchmarks, cfg):
    window = dict(gen_lo=cfg["read_window"]["gen_lo"], gen_hi=cfg["read_window"]["gen_hi"], prompt_hi=cfg["read_window"]["prompt_hi"])
    fold, n_folds = cfg["select_fold"], cfg["n_folds"]
    benchmarks = list(benchmarks)
    keep = lambda items: [(r, k) for r, k in items if traces[r]["bench"] in benchmarks]
    signal = signal_rows(traces, tuple(cfg["signal_roles"]))
    routed = [r for r in signal if is_routed(traces, r)]
    unrouted = [r for r in signal if not is_routed(traces, r)]
    select = lambda r: in_fold(r, fold, n_folds)
    rng = np.random.default_rng(cfg["list_seed"])
    unrouted_select_perm = permuted_per_benchmark(traces, [r for r in unrouted if select(r)], rng, window)
    unrouted_fit_perm = permuted_per_benchmark(traces, [r for r in unrouted if not select(r)], rng, window)
    lists = dict(
        fit=keep(one_read_each(traces, [r for r in routed if not select(r)], window)),
        fit_multi=keep(several_reads_each(traces, [r for r in routed if not select(r)], cfg["teacher_reads_per_row"], window)),
        select=keep(one_read_each(traces, [r for r in routed if select(r)], window)),
        unrouted_select=first_per_benchmark(unrouted_select_perm, cfg["unrouted_select_per_benchmark"], benchmarks),
    )
    n_first = cfg["unrouted_fit_per_benchmark"]
    unrouted_fit = first_per_benchmark(unrouted_fit_perm, n_first, benchmarks)
    continuations = {b: [unrouted_fit_perm[b][0][i] for i in unrouted_fit_perm[b][1][n_first:]] for b in benchmarks if b in unrouted_fit_perm}
    unrouted_fit += round_robin_continuation(continuations, benchmarks, max(0, cfg["unrouted_fit_total"] - len(unrouted_fit)))
    lists["unrouted_fit"] = sorted(unrouted_fit, key=lambda x: (traces[x[0]]["bench"], x[0]))
    lists["key_routed"] = lists["fit"] + lists["select"]
    lists["key_unrouted"] = lists["unrouted_fit"] + lists["unrouted_select"]
    return lists


def reads_for_rows(traces, rows_or_items, n_reads, min_tokens):
    row_ids = []
    for r, _ in rows_or_items:
        if r not in row_ids:
            row_ids.append(r)
    out = []
    for r in row_ids:
        if r not in traces:
            continue
        samples = sorted(int(k) for k, s in traces[r]["samples"].items() if s.get("finish") == "stop" and s.get("reasoning") and int(s["tokens"]) >= min_tokens)
        out += [(r, k) for k in samples[:n_reads]]
    return out


def counts_from_total(total, benchmarks):
    base, rest = divmod(int(total), len(benchmarks))
    return {b: base + (1 if i < rest else 0) for i, b in enumerate(benchmarks)}


def key_rows(traces, benchmarks, excluded_rows, cfg):
    window = cfg["key_window"]
    rng = np.random.default_rng(cfg["key_seed"])
    by_bench = {}
    for r in sorted(traces):
        record = traces[r]
        if record["role"] not in tuple(cfg["signal_roles"]):
            continue
        samples = [int(k) for k, s in record["samples"].items()
                   if window["gen_lo"] <= int(s["tokens"]) <= window["gen_hi"] and int(s["prompt_tokens"]) <= window["prompt_hi"] and s.get("reasoning") and s.get("finish") == "stop"]
        if samples:
            by_bench.setdefault(record["bench"], []).append((r, sorted(samples)))
    first, permutation = {}, {}
    n_first = cfg["key_first_per_benchmark"]
    for bench in sorted(by_bench):
        candidates = by_bench[bench]
        order = rng.permutation(len(candidates))
        permutation[bench] = [candidates[i][0] for i in order]
        first[bench] = []
        for i in sorted(order[:n_first]):
            row, samples = candidates[i]
            first[bench].append(row)
            rng.integers(len(samples))
    taken_first = {r for b in first for r in first[b]}
    chosen, shortfall = [], 0
    targets = counts_from_total(cfg["key_rows_total"], benchmarks)
    if n_first > min(targets.values()):
        raise fp.InputRefused(f"procedure_edit.lists.key_first_per_benchmark ({n_first}) is larger than the smallest per-benchmark share of "
                              f"key_rows_total ({min(targets.values())}); lower the first or raise the total")
    for bench in benchmarks:
        target = targets[bench]
        kept = first.get(bench, [])
        need = target - len(kept) + shortfall
        continuation = [r for r in permutation.get(bench, [])[n_first:] if r not in excluded_rows and r not in taken_first]
        added = continuation[:need]
        shortfall = need - len(added)
        chosen += kept + added
    if shortfall:
        raise fp.InputRefused(f"the teacher traces hold {shortfall} key rows too few for procedure_edit.lists.key_rows_total; lower it or read more teacher samples")
    items = [(r, pick_sample(traces, r, window["gen_lo"], window["gen_hi"], window["prompt_hi"])) for r in chosen]
    assert all(k is not None for _, k in items)
    return items, sorted(taken_first)


def exclusion_rows(traces, first_key_rows, per_benchmark, cfg):
    window = cfg["key_window"]
    rest = {r: v for r, v in traces.items() if r not in set(first_key_rows)}
    rng = np.random.default_rng(cfg["key_exclusion_seed"])
    by_bench = {}
    for r in sorted(rest):
        record = rest[r]
        if record["role"] not in tuple(cfg["signal_roles"]):
            continue
        samples = [int(k) for k, s in record["samples"].items()
                   if window["gen_lo"] <= int(s["tokens"]) <= cfg["key_exclusion_gen_hi"] and int(s["prompt_tokens"]) <= window["prompt_hi"] and s.get("reasoning") and s.get("finish") == "stop"]
        if samples:
            by_bench.setdefault(record["bench"], []).append((r, sorted(samples)))
    out = []
    for bench in sorted(by_bench):
        candidates = by_bench[bench]
        order = rng.permutation(len(candidates))[:per_benchmark]
        for i in sorted(order):
            out.append(candidates[i][0])
            rng.integers(len(candidates[i][1]))
    return out


def key_exclusions(traces, first_key_rows, select_items, cfg):
    excluded = {r for r, _ in select_items}
    for per_benchmark in cfg["key_exclusion_draws"]:
        excluded |= set(exclusion_rows(traces, first_key_rows, per_benchmark, cfg))
    return excluded
