import json
import os
import sys

import numpy as np

from basersi import config as C
from basersi import fingerprint as fp
from basersi.data import rows as RW
from basersi.data.rows import ROLE_FIELD
from basersi.engine.request import verify_images
from basersi.export.patch import copy_resolved, patch_folder, spec_inputs
from basersi.procedure_edit import evaluate, fit, gap_capture, gate, key, neurons, rows, select, site, spec
from basersi.procedure_edit.lab import Lab, read_items, write_items

TRACE_SETTINGS = ("min_image_tokens", "max_image_tokens", "temperature")
TRACE_ROW_FIELDS = ("bench", "kind", "answer", "prompt", "image_paths")
COMPARED_FIELDS = (ROLE_FIELD, "bench", "kind", "answer", "prompt")
SHOWN_IDS = 5
SIGNAL_ROLES = RW.SIGNAL_ROLES
WORK_KEY = "procedure_edit.work_dir"


def development_rows(rows_path):
    out = {}
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


LINE_SEARCH_FAILED = ("no step of procedure_edit.strength.steps has a finite held-out divergence on this build's replays, so there is no spec to export; "
                      "no setting turns this into a pass (the table and the replay folder are named here)")


class RowsRefused(C.Refused, ValueError):
    pass


def validated_development(rows_path):
    RW.validate(rows_path)
    return development_rows(rows_path)


def id_list(ids):
    ids = sorted(ids)
    return f"{len(ids)} ({', '.join(repr(i) for i in ids[:SHOWN_IDS])}{', ...' if len(ids) > SHOWN_IDS else ''})"


def row_problems(ids, development, benchmarks, signal_roles, records=None):
    foreign, unknown, validation, differing = [], [], [], []
    for r in ids:
        row = development.get(r)
        claimed = (records or {}).get(r) if isinstance((records or {}).get(r), dict) else {}
        bench = row["bench"] if row else claimed.get("bench")
        if bench is not None and bench not in benchmarks:
            foreign.append(r)
        elif row is None:
            unknown.append(r)
        elif row[ROLE_FIELD] not in signal_roles:
            validation.append(r)
        elif any(k in claimed and claimed[k] != row.get(k) for k in COMPARED_FIELDS):
            differing.append(r)
    return [(what, found) for what, found in (
        (f"rows of benchmarks outside the configured {list(benchmarks)} (no step reads them: pass a copy of the file without them)", foreign),
        ("ids that are not in the development rows", unknown),
        (f"validation rows (development role not in {list(signal_roles)})", validation),
        (f"records whose {'/'.join(COMPARED_FIELDS)} differ from the development row of the same id", differing)) if found]


def refuse_rows(problems, source):
    if problems:
        raise RowsRefused(f"{source}: " + "; ".join(f"{what}: {id_list(found)}" for what, found in problems)
                          + ". The Procedure Edit takes only development signal rows of the configured benchmarks; "
                            "roles and row content come from the validated development rows, never from this file")


def check_row_ids(ids, development, benchmarks, signal_roles, source):
    refuse_rows(row_problems(list(ids), development, benchmarks, signal_roles), source)
    return list(ids)


def checked_traces(traces, development, benchmarks, signal_roles, source):
    if not isinstance(traces, dict) or not all(isinstance(v, dict) and isinstance(v.get("samples"), dict) for v in traces.values()):
        raise RowsRefused(f"{source} is not a traces file (a JSON object: row id -> record with a samples object)")
    refuse_rows(row_problems(sorted(traces), development, benchmarks, signal_roles, records=traces), source)
    return {r: dict(v, **{k: development[r].get(k) for k in TRACE_ROW_FIELDS}, **{ROLE_FIELD: development[r][ROLE_FIELD]}) for r, v in traces.items()}


def check_records(directory, items, development, benchmarks, signal_roles):
    check_row_ids({r for r, _ in items}, development, benchmarks, signal_roles, f"the item list of {directory}")
    expected = {gap_capture.record_name(r, k): (r, k) for r, k in items}
    present = sorted(n for n in os.listdir(directory) if n.endswith(".npz"))
    extra = [n for n in present if n not in expected]
    if extra:
        raise RowsRefused(f"{directory} holds {id_list(extra)} records that are not in its item list; the fit and the line search read every record "
                          f"of the folder, so it must hold only the listed reads: {fp.way_forward([directory], WORK_KEY)}")
    wrong = []
    for name in present:
        with np.load(os.path.join(directory, name)) as z:
            if (str(z["row"]), int(z["sample"])) != expected[name]:
                wrong.append(name)
    if wrong:
        raise RowsRefused(f"{directory}: {id_list(wrong)} records hold another row or read than their name says: {fp.way_forward([directory], WORK_KEY)}")
    return directory


def device_type(device):
    return str(device).split(":")[0]


def lab_settings():
    engine = C.load("engine")
    return dict(reasoning_effort=engine["reasoning_effort"], image=engine["image"])


def trace_settings(request):
    return {k: (request.get("settings") or {}).get(k) for k in TRACE_SETTINGS}


def trace_templates(development, row_ids, request):
    settings = trace_settings(request)
    prefix_rows = request.get("prefix_rows")
    out = {}
    for row_id in row_ids:
        row = development[row_id]
        record = {k: row.get(k) for k in TRACE_ROW_FIELDS}
        record["role"] = row[ROLE_FIELD]
        routed = request.get("prefix") and (prefix_rows is None or row_id in set(prefix_rows))
        record["prefix"] = request["prefix"] if routed else None
        record["settings"] = dict(settings)
        out[row_id] = record
    return out


def collect_reads(reads_path, request, keep_rows):
    samples_by_row = {}
    with open(reads_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("error") or r.get("finish") == "abort" or r.get("row") not in keep_rows:
                continue
            assert r.get("req") == request, (r.get("row"), r.get("req"))
            record = dict(reasoning=r.get("reasoning") or "", content=r.get("content") or "", tokens=r["tokens"], prompt_tokens=r.get("prompt_tokens"),
                          ok=bool(r["ok"]), finish=r.get("finish"), pred=r.get("pred"), seed=r.get("seed"), image_tokens=r.get("image_tokens"))
            existing = samples_by_row.setdefault(r["row"], {})
            if str(r["sample"]) in existing:
                assert existing[str(r["sample"])]["tokens"] == record["tokens"]
            existing[str(r["sample"])] = record
    return samples_by_row


def traces_from_reads(reads_path, templates, request, out_path):
    samples_by_row = collect_reads(reads_path, request, set(templates))
    out = {}
    for row_id, record in templates.items():
        if row_id in samples_by_row:
            entry = {k: v for k, v in record.items() if k != "samples"}
            entry["samples"] = dict(sorted(samples_by_row[row_id].items(), key=lambda kv: int(kv[0])))
            out[row_id] = entry
    return fp.write_json_atomic(out_path, out)


def sample_range(bounds):
    return list(range(int(bounds[0]), int(bounds[1]) + 1))


def records_complete(out_dir, items):
    return all(os.path.exists(os.path.join(out_dir, gap_capture.record_name(r, k))) for r, k in items)


def check_names(cfg):
    C.check_keys(cfg, C.load("procedure_edit"), "procedure_edit", prefix=["procedure_edit"])
    C.check_strength(cfg)


class Compiler:
    def __init__(self, cfg, benchmarks, model_dir, work_dir, rows_path, read_rows, entry=None, teacher_traces=None):
        check_names(cfg)
        if int(cfg["rounds"]) < 2:
            raise C.ConfigError(f"procedure_edit.rounds must be at least 2 (round 2 is the first round fitted on the route key), got {cfg['rounds']}")
        fp.require_stock(model_dir, "the Procedure Edit (it is fitted on the stock weights and stacked on the View Edit only at export)")
        self.cfg, self.benchmarks = cfg, list(benchmarks)
        self.model_dir, self.work, self.rows_path = model_dir, work_dir, rows_path
        self.read_rows, self.entry = read_rows, entry
        self.lab = self.lab_traces = None
        self.identities, self.checked = {}, {}
        self.signal_roles = SIGNAL_ROLES
        self.device_type = device_type(cfg["device"])
        self.development = validated_development(rows_path)
        verify_images([self.development[r] for r in sorted(self.development) if self.development[r][ROLE_FIELD] in SIGNAL_ROLES and self.development[r]["bench"] in self.benchmarks])
        if teacher_traces:
            self.traces_at(teacher_traces)
        os.makedirs(work_dir, exist_ok=True)
        self.teacher_traces_path = teacher_traces or self.build_teacher_traces()
        self.teacher = self.traces_at(self.teacher_traces_path)
        self.teacher_id = fp.file_sha256(self.teacher_traces_path)
        self.list_cfg = dict(cfg["lists"], signal_roles=list(SIGNAL_ROLES), select_fold=cfg["select_fold"], n_folds=cfg["n_folds"])

    def path(self, *parts):
        return os.path.join(self.work, *parts)

    def traces_at(self, path):
        sha = fp.file_sha256(path)
        if self.checked.get(path, (None,))[0] != sha:
            traces = fp.given_json(path, "the traces file", "a traces JSON object (row id -> record with a samples object)")
            self.checked[path] = (sha, checked_traces(traces, self.development, self.benchmarks, self.signal_roles, path))
        return self.checked[path][1]

    def check_records(self, directory, items):
        return check_records(directory, items, self.development, self.benchmarks, self.signal_roles)

    def reuse(self, path, inputs, reads_rows=True):
        return fp.reuse(path, inputs, key=WORK_KEY, reads_rows=reads_rows)

    def identity(self, model_dir):
        if model_dir not in self.identities:
            self.identities[model_dir] = fp.checkpoint_identity(model_dir)
        return self.identities[model_dir]

    def release_lab(self):
        if self.lab is not None:
            import gc
            import torch
            self.lab = self.lab_traces = None
            gc.collect()
            torch.cuda.empty_cache()

    def read(self, model_dir, row_ids, samples, request, reads_path):
        self.release_lab()
        return self.read_rows(model_dir, row_ids, samples, request, reads_path, rows_path=self.rows_path, **self.cfg.get("serving", {}))

    def teacher_request(self):
        routed = check_row_ids(self.entry["routed_signal_rows"], self.development, self.benchmarks, self.signal_roles, "the routed rows of the procedure entry")
        return {"settings": dict(self.entry["view_settings"]), "prefix": self.entry["procedure"], "prefix_rows": routed}

    def build_teacher_traces(self):
        p = self.path("traces", "teacher.json")
        assert self.entry is not None, "no procedure entry and no teacher traces"
        development = self.development
        row_ids = sorted(r for r, v in development.items() if v[ROLE_FIELD] in self.signal_roles and v["bench"] in self.benchmarks)
        request = self.teacher_request()
        samples = sample_range(self.cfg["teacher_samples"])
        inputs = dict(model=self.identity(self.model_dir), rows=fp.file_sha256(self.rows_path), row_ids=row_ids, samples=samples, request=request,
                      read_settings=C.read_settings(direct=True))
        reads_path = self.path("traces", "teacher_reads.jsonl")
        self.reuse(reads_path, inputs)
        if self.reuse(p, inputs):
            return p
        reads = self.read(self.model_dir, row_ids, samples, request, reads_path)
        return traces_from_reads(reads, trace_templates(development, row_ids, request), request, p)

    def lab_for(self, traces_path, f32, max_reason):
        if self.lab is None:
            self.lab = Lab(self.model_dir, self.traces_at(traces_path), device=self.cfg["device"], tail_positions=self.cfg["tail_positions"], f32=f32,
                           base_drops_prefix=True, max_reason=max_reason, memory_fraction=self.cfg.get("memory_fraction"))
            self.lab_traces = traces_path
        if self.lab_traces != traces_path:
            self.lab.traces = self.traces_at(traces_path)
            self.lab_traces = traces_path
        self.lab.f32, self.lab.max_reason = f32, max_reason
        return self.lab

    def item_list(self, name, items, inputs):
        p = self.path("rows", f"{name}.txt")
        if not self.reuse(p, inputs):
            write_items(p, items)
        held = read_items(p)
        check_row_ids({r for r, _ in held}, self.development, self.benchmarks, self.signal_roles, p)
        return held

    def compile_rows(self):
        inputs = dict(teacher=self.teacher_id, lists=self.list_cfg, benchmarks=self.benchmarks)
        lists = rows.compile_lists(self.teacher, self.benchmarks, self.list_cfg)
        out = {name: self.item_list(name, items, inputs) for name, items in lists.items()}
        _, first = rows.key_rows(self.teacher, self.benchmarks, set(), self.list_cfg)
        excluded = rows.key_exclusions(self.teacher, first, out["select"], self.list_cfg)
        key_items, _ = rows.key_rows(self.teacher, self.benchmarks, excluded, self.list_cfg)
        out["position_key"] = self.item_list("position_key", key_items, inputs)
        self.lists = out
        return out

    def measure_site(self):
        cfg, capture = self.cfg["site"], self.cfg["capture"]
        per_row, min_tokens = capture["student_reads_per_row"], capture["student_min_tokens"]
        base_traces = self.floor_traces("round2_base", self.model_dir, self.identity(self.model_dir), self.cfg["base_floor_samples"])
        items = self.item_list("site_own", rows.reads_for_rows(self.traces_at(base_traces), self.lists["select"], per_row, min_tokens),
                               dict(traces=fp.file_sha256(base_traces), select=self.lists["select"], per_row=per_row, min_tokens=min_tokens))
        out = self.path("site", "own")
        self.reuse(out, dict(model=self.identity(self.model_dir), traces=fp.file_sha256(base_traces), items=items, device_type=self.device_type,
                           f32=cfg["f32"], max_reason=cfg["max_reason"], tail_positions=self.cfg["tail_positions"], lab=lab_settings()))
        os.makedirs(out, exist_ok=True)
        if not records_complete(out, items):
            lab = self.lab_for(base_traces, cfg["f32"], cfg["max_reason"])
            site.measure(lab, items, out)
        self.check_records(out, items)
        summary_path = self.path("site", "linear_fraction.json")
        if not self.reuse(summary_path, dict(own=fp.of(out))):
            fp.write_json_atomic(summary_path, site.linear_fraction(site.load_records(out)), indent=1)
        return dict(fp.read_json(summary_path), file=summary_path, records=out)

    def select_neurons(self):
        cfg = self.cfg["neurons"]
        p = self.path("neurons_free.json")
        used = cfg.get("used_neurons")
        inputs = dict(model=self.identity(self.model_dir), teacher=self.teacher_id, fit=self.lists["fit"], unrouted_select=self.lists["unrouted_select"], device_type=self.device_type,
                      neurons={k: v for k, v in cfg.items() if k != "used_neurons"}, used=fp.file_sha256(used) if used else None,
                      tail_positions=self.cfg["tail_positions"], lab=lab_settings())
        if not self.reuse(p, inputs):
            lab = self.lab_for(self.teacher_traces_path, cfg["f32"], 0)
            neurons.select_free_neurons(lab, self.lists["fit"], self.lists["unrouted_select"], used, p, cfg)
        return p

    def capture_keys(self, name, items, drop_prefix):
        cfg = self.cfg["position_keys"]
        out = self.path("key_captures", name)
        self.reuse(out, dict(model=self.identity(self.model_dir), traces=self.teacher_id, items=items, positions=cfg["positions"], seed=cfg["seed"], device_type=self.device_type,
                           f32=cfg["f32"], drop_prefix=drop_prefix, tail_positions=self.cfg["tail_positions"], lab=lab_settings()))
        os.makedirs(out, exist_ok=True)
        if not records_complete(out, items):
            lab = self.lab_for(self.teacher_traces_path, cfg["f32"], 0)
            lab.base_drops_prefix = drop_prefix
            try:
                key.capture_unit_inputs(lab, items, out, cfg["positions"], cfg["seed"])
            finally:
                lab.base_drops_prefix = True
        return self.check_records(out, items)

    def position_keys(self):
        cfg = self.cfg["position_keys"]
        d = self.capture_keys("position", self.lists["position_key"], cfg["drop_prefix"])
        p = self.path("keys", "position_keys.npz")
        if not self.reuse(p, dict(captures=fp.of(d), ridge=cfg["ridge"], select_fold=self.cfg["select_fold"], n_folds=self.cfg["n_folds"], device_type=self.device_type)):
            arrays = key.fit_position_keys(key.load_unit_inputs(d), {}, cfg["ridge"], self.cfg["select_fold"], self.cfg["n_folds"], self.cfg["device"])
            fit.save_arrays(p, arrays)
        return p

    def capture_round(self, name, traces_path, items, round_cfg):
        capture = self.cfg["capture"]
        out = self.path("captures", name)
        self.reuse(out, dict(model=self.identity(self.model_dir), traces=fp.file_sha256(traces_path), items=items, f32=capture["f32"], seed=capture["seed"], device_type=self.device_type,
                           round=round_cfg, tail_positions=self.cfg["tail_positions"], lab=lab_settings()))
        os.makedirs(os.path.join(out, "samp"), exist_ok=True)
        if not records_complete(out, items):
            lab = self.lab_for(traces_path, capture["f32"], round_cfg["max_reason"])
            for shard in range(round_cfg["shards"]):
                gap_capture.capture_gaps(lab, items, out, round_cfg["sampled_positions"], capture["seed"], shard=shard, n_shards=round_cfg["shards"])
        return self.check_records(out, items)

    def fit_round(self, name, aggregate, base_arrays_path):
        cfg = self.cfg["fit"]
        p = self.path("cells", f"{name}.npz")
        inputs = dict(aggregate=[[fp.of(d), tag] for d, tag in aggregate], base=fp.file_sha256(base_arrays_path), device_type=self.device_type,
                      fit={k: v for k, v in cfg.items() if k != "n_threads"}, val_fold=self.cfg["fit_val_fold"], n_folds=self.cfg["n_folds"],
                      tail_positions=self.cfg["tail_positions"], unit_gate_scale=self.cfg["unit_gate_scale"])
        if not self.reuse(p, inputs):
            arrays, report = fit.fit_cell(fit.aggregate_records(aggregate), fit.load_arrays(base_arrays_path), cfg["rank"], cfg["ridge_grid"],
                                          self.cfg["fit_val_fold"], self.cfg["n_folds"], self.cfg["device"], cfg["seed"], self.cfg["tail_positions"],
                                          self.cfg["unit_gate_scale"], cfg["oversampling"], cfg["power_iterations"], cfg["decomposition"], cfg["n_threads"])
            fp.write_json_atomic(self.path("cells", f"{name}.json"), report, indent=1)
            fit.save_arrays(p, arrays)
        return p

    def route_key_arrays(self, round1_arrays):
        cfg = self.cfg["route_key"]
        routed = self.capture_keys("routed", self.lists["key_routed"], cfg["drop_prefix"])
        unrouted = self.capture_keys("unrouted", self.lists["key_unrouted"], cfg["drop_prefix"])
        p = self.path("cells", "round1_route_key.npz")
        inputs = dict(routed=fp.of(routed), unrouted=fp.of(unrouted), round1=fp.file_sha256(round1_arrays), ridge=cfg["ridge"], device_type=self.device_type,
                      select_fold=self.cfg["select_fold"], n_folds=self.cfg["n_folds"])
        if not self.reuse(p, inputs):
            arrays, counts = key.fit_route_key(key.load_unit_inputs(routed), key.load_unit_inputs(unrouted), fit.load_arrays(round1_arrays),
                                               cfg["ridge"], self.cfg["select_fold"], self.cfg["n_folds"], self.cfg["device"])
            fp.write_json_atomic(self.path("cells", "round1_route_key.json"), counts, indent=1)
            fit.save_arrays(p, arrays)
        return p

    def routed_signal_rows(self):
        return [r for r in rows.signal_rows(self.teacher, self.signal_roles) if rows.is_routed(self.teacher, r) and self.teacher[r]["bench"] in self.benchmarks]

    def floor_traces(self, name, model_dir, model_id, samples):
        p = self.path("traces", f"{name}.json")
        reads_path = self.path("traces", f"{name}_reads.jsonl")
        routed = self.routed_signal_rows()
        request = self.cfg["floor_request"]
        samples = sample_range(samples)
        inputs = dict(model=model_id, rows=fp.file_sha256(self.rows_path), teacher=self.teacher_id, row_ids=routed, samples=samples, request=request,
                      read_settings=C.read_settings(direct=True))
        self.reuse(reads_path, inputs)
        if not self.reuse(p, inputs):
            reads = self.read(model_dir, routed, samples, request, reads_path)
            settings = trace_settings(request)
            traces_from_reads(reads, {r: dict(self.teacher[r], settings=dict(settings)) for r in routed}, request, p)
        return p

    def student_cell(self, round_index, arrays_path, own_traces):
        student, strength, capture = self.cfg["student"], self.cfg["strength"], self.cfg["capture"]
        cell = spec.parse_cell(student["cell"])
        if not student["line_search"]:
            return student["cell"], dict(step=cell["step"], chosen_by="preset")
        steps = [float(s) for s in strength["steps"]]
        blocks_units = f"{cell['lo']}-{cell['hi']}:{cell['rank']}"
        replay_cells = [f"{select.candidate_name(s)}={blocks_units}:{s:g}:gate={cell['gate']}" + ("" if cell["bias"] else ",nobias") for s in steps]
        name = f"round{round_index}"
        items = self.item_list(f"strength_{name}", rows.reads_for_rows(self.traces_at(own_traces), self.lists["select"],
                                                                       capture["student_reads_per_row"], capture["student_min_tokens"]),
                               dict(traces=fp.file_sha256(own_traces), select=self.lists["select"], per_row=capture["student_reads_per_row"],
                                    min_tokens=capture["student_min_tokens"]))
        own_dir = self.replay(name, own_traces, items, replay_cells, arrays_path)
        table = select.measure_steps(steps, evaluate.load_replay(own_dir))
        best = select.line_search(table)
        table_path = self.path("strength", f"{name}_table.json")
        if not self.reuse(table_path, dict(own=fp.of(own_dir), steps=steps)):
            fp.write_json_atomic(table_path, dict(steps=table, chosen=best and best["name"]), indent=1)
        if best is None:
            print(f"[basersi] round {round_index}: no step of procedure_edit.strength.steps has a finite held-out divergence ({table_path}); "
                  f"the student keeps the step of procedure_edit.student.cell", file=sys.stderr, flush=True)
            return student["cell"], dict(step=cell["step"], chosen_by="preset", table=table_path)
        step = float(best["step"])
        return f"{blocks_units}:{step:g}:{cell['gate']}" + ("" if cell["bias"] else ":nobias"), dict(step=step, chosen_by="line_search", table=table_path)

    def student_checkpoint(self, round_index, arrays_path, neurons_path, cell):
        spec_path = self.path("cells", f"round{round_index}_student_spec.json")
        if not self.reuse(spec_path, dict(cell=cell, arrays=fp.file_sha256(arrays_path), neurons=fp.file_sha256(neurons_path),
                                        unit_gate_scale=self.cfg["unit_gate_scale"])):
            spec.write_spec(spec_path, cell, fit.array_keys(arrays_path), spec.load_free(neurons_path), arrays_path, self.cfg["unit_gate_scale"])
        folder = self.path("students", f"round{round_index}")
        report_path = folder + ".json"
        inputs = spec_inputs(self.model_dir, spec_path)
        if self.reuse(report_path, inputs, reads_rows=False):
            held = fp.last_edit(folder, "student")
            if held is None or held["inputs_fingerprint"] != fp.of(report_path):
                raise fp.StaleOutput(f"{folder} does not hold the student copy its report {report_path} describes (gone or edited): "
                                     f"{fp.way_forward([folder, report_path, fp.sidecar(report_path)], WORK_KEY)}")
            return folder, fp.of(report_path)
        if os.path.lexists(folder):
            edit = fp.last_edit(folder, "student")
            matched = fp.match(edit["inputs_fingerprint"], edit["report"].get("inputs"), inputs, folder, reads_rows=False) if edit else None
            if not matched:
                raise fp.StaleOutput(f"{folder} exists without its report {report_path} and holds {'another student copy' if edit else f'no {fp.MARKER}'}: "
                                     f"{fp.way_forward([folder], WORK_KEY)}")
            report = fp.with_other_code_record(dict(edit["report"], recovered_from=os.path.join(folder, fp.MARKER)), matched)
        else:
            partial = folder + ".partial"
            set_aside = fp.set_aside(partial) if os.path.lexists(partial) else None
            copy_resolved(self.model_dir, partial)
            report = patch_folder(partial, self.model_dir, spec_path, stage="student", extra=dict(folder=folder))
            os.rename(partial, folder)
            report["set_aside"] = set_aside
        fp.write_json_atomic(report_path, report, indent=1)
        return folder, fp.of(report_path)

    def compile(self):
        cfg = self.cfg
        capture = cfg["capture"]
        min_tokens, per_row = capture["student_min_tokens"], capture["student_reads_per_row"]
        self.compile_rows()
        site_record = self.measure_site()
        neurons_path = self.select_neurons()
        keys_path = self.position_keys()
        teacher_dir = self.capture_round("round1_teacher", self.teacher_traces_path, self.lists["fit_multi"], capture["teacher"])
        round_arrays = [self.fit_round("round1", [(teacher_dir, None)], keys_path)]
        route_arrays = self.route_key_arrays(round_arrays[0])
        aggregate = [(teacher_dir, "t")]
        base_traces = self.floor_traces("round2_base", self.model_dir, self.identity(self.model_dir), cfg["base_floor_samples"])
        reads_inputs = dict(fit=self.lists["fit"], per_row=per_row, min_tokens=min_tokens)
        base_items = self.item_list("round2_base_fit", rows.reads_for_rows(self.traces_at(base_traces), self.lists["fit"], per_row, min_tokens),
                                    dict(reads_inputs, traces=fp.file_sha256(base_traces)))
        aggregate.append((self.capture_round("round2_base", base_traces, base_items, capture["base"]), "f"))
        round_arrays.append(self.fit_round("round2", aggregate, route_arrays))
        student_traces, student_steps = None, {}
        for r in range(3, cfg["rounds"] + 1):
            cell, student_steps[f"round{r - 1}"] = self.student_cell(r - 1, round_arrays[-1], student_traces or base_traces)
            student, student_id = self.student_checkpoint(r - 1, round_arrays[-1], neurons_path, cell)
            student_traces = self.floor_traces(f"round{r}_student", student, student_id, cfg["student"]["samples"])
            items = self.item_list(f"round{r}_student_fit", rows.reads_for_rows(self.traces_at(student_traces), self.lists["fit"], per_row, min_tokens),
                                   dict(reads_inputs, traces=fp.file_sha256(student_traces)))
            aggregate.append((self.capture_round(f"round{r}_student", student_traces, items, capture["student"]), f"l{r}"))
            round_arrays.append(self.fit_round(f"round{r}", aggregate, route_arrays))
        gated_path = self.path("cells", "final_gated.npz")
        if not self.reuse(gated_path, dict(arrays=fp.file_sha256(round_arrays[-1]), key=cfg["key"], rank=cfg["fit"]["rank"],
                                         unit_gate_scale=cfg["unit_gate_scale"])):
            fit.save_arrays(gated_path, gate.add_key(fit.load_arrays(round_arrays[-1]), cfg["key"]["slope"], cfg["key"]["threshold"], cfg["fit"]["rank"],
                                                     cfg["unit_gate_scale"]))
        last_block = gate.block_count(fit.array_keys(gated_path), "k") - 1
        choice = self.choose(gated_path, student_traces or base_traces, last_block)
        if choice.get("failed"):
            return dict(spec=None, line_search_failed=choice["failed"], site=site_record, arrays=gated_path, neurons=neurons_path, position_keys=keys_path,
                        route_key_arrays=route_arrays, round_arrays=round_arrays, teacher_traces=self.teacher_traces_path, student_steps=student_steps,
                        strength=dict(step=None, chosen_by="line_search", at_grid_edge=None, table=choice["table"]))
        cell_text = select.spec_cell_text(choice["step"], cfg["fit"]["rank"], last_block)
        spec_path = self.path("final_spec.json")
        if not self.reuse(spec_path, dict(cell=cell_text, arrays=fp.file_sha256(gated_path), neurons=fp.file_sha256(neurons_path),
                                        unit_gate_scale=cfg["unit_gate_scale"])):
            spec.write_spec(spec_path, cell_text, fit.array_keys(gated_path), spec.load_free(neurons_path), gated_path, cfg["unit_gate_scale"])
        written = spec.describe_file(spec_path)
        if written["cell"] != cell_text:
            raise fp.StaleOutput(f"{spec_path} holds the cell {written['cell']}, not {cell_text}: {fp.way_forward([spec_path, fp.sidecar(spec_path)], WORK_KEY)}")
        return dict(spec=spec_path, spec_sha256=fp.file_sha256(spec_path), site=site_record, arrays=gated_path, neurons=neurons_path, position_keys=keys_path,
                    route_key_arrays=route_arrays, round_arrays=round_arrays, teacher_traces=self.teacher_traces_path, cell=written["cell"],
                    student_steps=student_steps, step=written["step"], strength=dict(step=written["step"], chosen_by=choice["chosen_by"], at_grid_edge=choice.get("at_grid_edge"),
                                                        table=choice.get("table")))

    def replay(self, name, traces_path, items, cells, gated_path):
        strength = self.cfg["strength"]
        out = self.path("strength", name)
        self.reuse(out, dict(model=self.identity(self.model_dir), traces=fp.file_sha256(traces_path), items=items, cells=cells, device_type=self.device_type,
                           arrays=fp.file_sha256(gated_path), f32=strength["f32"], max_reason=strength["max_reason"], unit_gate_scale=self.cfg["unit_gate_scale"],
                           lab=lab_settings()))
        os.makedirs(out, exist_ok=True)
        if not records_complete(out, items):
            lab = self.lab_for(traces_path, strength["f32"], strength["max_reason"])
            evaluate.replay(lab, fit.load_arrays(gated_path), cells, items, out, self.cfg["unit_gate_scale"])
        return self.check_records(out, items)

    def choose(self, gated_path, select_traces, last_block):
        cfg = self.cfg
        strength = cfg["strength"]
        if strength["step"] is not None:
            return dict(step=float(strength["step"]), chosen_by="preset")
        steps = [float(s) for s in strength["steps"]]
        cells = [select.replay_cell_text(s, cfg["fit"]["rank"], last_block) for s in steps]
        capture = cfg["capture"]
        own_items = self.item_list("strength_own", rows.reads_for_rows(self.traces_at(select_traces), self.lists["select"],
                                                                        capture["student_reads_per_row"], capture["student_min_tokens"]),
                                   dict(traces=fp.file_sha256(select_traces), select=self.lists["select"], per_row=capture["student_reads_per_row"],
                                        min_tokens=capture["student_min_tokens"]))
        own_dir = self.replay("own", select_traces, own_items, cells, gated_path)
        table = select.measure_steps(steps, evaluate.load_replay(own_dir))
        best = select.line_search(table)
        at_edge = None if best is None else best["step"] in (min(steps), max(steps))
        table_path = self.path("strength", "table.json")
        if not self.reuse(table_path, dict(own=fp.of(own_dir), steps=steps)):
            fp.write_json_atomic(table_path, dict(steps=table, chosen=best and best["name"], at_grid_edge=at_edge), indent=1)
        if best is None:
            return dict(failed=dict(reason=LINE_SEARCH_FAILED, table=table_path, replays=dict(own=own_dir), candidates=[r["name"] for r in table]), table=table_path)
        if at_edge:
            print(f"[basersi] the line search took step {best['step']:g}, an end of procedure_edit.strength.steps: a lower divergence may lie outside these steps "
                  f"({table_path})", file=sys.stderr, flush=True)
        return dict(step=float(best["step"]), chosen_by="line_search", at_grid_edge=at_edge, table=table_path)
