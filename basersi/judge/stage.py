import os

from basersi import config as C
from basersi import fingerprint as FP
from basersi.judge import candidate_reads as CR
from basersi.judge import exports as EX
from basersi.judge import reads as R
from basersi.judge import stage_entries, stage_procedure_edit, stage_view_edit

PARTS = ("entries", "view_edit", "procedure_edit")


def judge_config(section):
    return C.load("judge", section.get("config_override"))


def procedure_entry(rows_path, export_dir, cfg, verdicts):
    rows = R.load_rows(rows_path)
    ex = EX.Exports(export_dir, cfg["exports"], cfg.get("sha256"))
    _, fired, ents = ex.tip(rows)
    wo, wo_rows = ex.entry_removals(rows)
    groups = stage_entries.entry_groups(cfg, wo_rows)
    special = [g for g, spec in groups.items() if verdicts.get(spec["label"]) == "ID_SPECIFIC" and len(spec["entries"]) == 1]
    chosen = cfg.get("procedure_entry_group")
    if chosen is None:
        if len(special) != 1:
            return dict(specialisation_groups=special, procedure_entry=None)
        chosen = special[0]
    if chosen not in groups:
        raise C.ConfigError(f"judge.procedure_entry_group {chosen!r} is not an entry group of this export; groups: {sorted(groups)}")
    index = groups[chosen]["entries"][0]
    entry = dict(ents[index])
    signal = set(R.rows_of(rows, R.SIGNAL_ROLES))
    entry["routed_signal_rows"] = sorted(r for r, f in fired.items() if index in f and r in signal)
    entry["index"] = index
    sources = ex.entry_sources().get(index) or []
    entry["sources"] = sources
    entry["source"] = sources[0] if len(sources) == 1 else None
    view = [e for e in ents if e.get("procedure") is None and e.get("settings")]
    entry["view_settings"] = dict(view[0]["settings"]) if view else {}
    return dict(specialisation_groups=special, procedure_entry=entry)


def read_candidate_exports(config, section, cfg, part, out_dir):
    outputs = config.get("outputs") or {}
    model_dir = section.get(f"{part}_checkpoint") or (outputs.get("view_edit" if part == "view_edit" else "export") or {}).get("checkpoint")
    if not model_dir:
        raise C.MissingInput(f"judge: no checkpoint for the {part} reads; run the {'view_edit' if part == 'view_edit' else 'export'} stage first or set judge.{part}_checkpoint")
    from basersi.export.stage import check_export_base
    base_check = check_export_base(model_dir, config["base_model"])
    samples = cfg["candidate_samples"][part]
    for role_key, roles in (("signal", R.SIGNAL_ROLES), ("judge", ("judge",))):
        key = f"{part}_{role_key}"
        spec = cfg["exports"][key]
        read_path = os.path.join(out_dir, "reads", f"{key}.jsonl")
        out_gz = os.path.join(out_dir, "exports", spec["file"])
        got = CR.read_candidate(model_dir, (outputs.get("data") or {}).get("development_rows") or os.path.join(config["data_dir"], C.load("benchmarks")["development"]["file"]), roles, samples[role_key], spec["arm_hint"], read_path, out_gz,
                                cards=cfg["candidate_read_cards"], port=int(cfg["candidate_read_port"]),
                                accept_other_code=bool(config.get(FP.OTHER_CODE_KEY)), allow_missing_reads=bool(cfg["exports"].get("allow_missing_reads")))
        base_check = dict(base_check, **{f"{key}_reads": dict(expected=got["expected_reads"], missing_or_errored=got["missing_reads"])})
        if got.get("read_by_other_code"):
            base_check = dict(base_check, **{f"{key}_read_by_other_code": got["read_by_other_code"]})
        spec["path"] = out_gz
    return base_check


def run(config):
    section = dict(config.get("judge") or {})
    cfg = judge_config(section)
    for k in ("procedure_edit_source_benchmark", "procedure_edit_entry_index", "procedure_entry_group", "candidate_read_port", "candidate_read_cards"):
        if section.get(k) is not None:
            cfg[k] = section[k]
    outputs = config.get("outputs") or {}
    rows_path = (outputs.get("data") or {}).get("development_rows") or os.path.join(config["data_dir"], C.load("benchmarks")["development"]["file"])
    if not os.path.exists(rows_path):
        raise C.MissingInput(f"judge: the development rows {rows_path} do not exist and no data step has run in this work_dir: run the data step first "
                         f"(pipeline --only data, or --from data)")
    export_dir = section.get("export_dir") or (outputs.get("loop") or {}).get("judge_exports")
    if not export_dir:
        raise C.MissingInput("judge: no export_dir: set judge.export_dir, or run the loop step in this work_dir first (its judge_exports are used)")
    out_dir = section.get("out_dir") or os.path.join(config["work_dir"], "judge")
    parts = list(section.get("parts") or PARTS)
    bad = [p for p in parts if p not in PARTS]
    if bad:
        raise C.ConfigError(f"judge: unknown parts {bad}; known {list(PARTS)}")
    EX.Exports(export_dir, cfg["exports"], cfg.get("sha256"))
    result = dict(outputs.get("judge") or {})
    result["out_dir"] = out_dir
    if "entries" in parts:
        res = stage_entries.run(rows_path, export_dir, out_dir, cfg)
        result.update(res)
        result.update(procedure_entry(rows_path, export_dir, cfg, res["entry_layers"]))
        entry = result.get("procedure_entry")
        if entry:
            cfg["procedure_edit_entry_index"] = entry["index"]
            if entry.get("source"):
                cfg["procedure_edit_source_benchmark"] = entry["source"]
    if "view_edit" in parts or "procedure_edit" in parts:
        for p in ("view_edit", "procedure_edit"):
            given = result.get("candidate_exports", {}).get(p)
            if given:
                for k, v in given.items():
                    cfg["exports"][k]["path"] = v
        for p in [x for x in ("view_edit", "procedure_edit") if x in parts]:
            if section.get("read_candidates", True):
                result.setdefault("candidate_base_check", {})[p] = read_candidate_exports(config, section, cfg, p, out_dir)
                result.setdefault("candidate_exports", {})[p] = {f"{p}_{r}": cfg["exports"][f"{p}_{r}"]["path"] for r in ("signal", "judge")}
        if section.get("read_candidates", True) and "procedure_edit" in parts and not result.get("candidate_exports", {}).get("view_edit"):
            raise C.MissingInput("judge procedure_edit: the View Edit reads are needed as the reference (run the view_edit part first)")
    if "view_edit" in parts:
        ctx = stage_view_edit.Context(rows_path, export_dir, out_dir, cfg)
        sig = stage_view_edit.run_signal(ctx)
        jud = stage_view_edit.run_judge(ctx)
        result["view_edit"] = dict(sig, **jud)
    if "procedure_edit" in parts:
        entry = result.get("procedure_entry") or {}
        if entry.get("index") is not None:
            cfg["procedure_edit_entry_index"] = entry["index"]
        if entry.get("source"):
            cfg["procedure_edit_source_benchmark"] = entry["source"]
        ctx = stage_procedure_edit.Context(rows_path, export_dir, out_dir, cfg)
        sig = stage_procedure_edit.run_signal(ctx)
        jud = stage_procedure_edit.run_judge(ctx)
        result["procedure_edit"] = dict(sig, **jud)
    return result
