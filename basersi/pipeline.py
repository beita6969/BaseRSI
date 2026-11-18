import copy
import datetime
import importlib
import json
import os
import sys

from basersi import config as C
from basersi import fingerprint as FP

STAGE_MODULES = {
    "data": "basersi.data.stage",
    "view_edit": "basersi.view_edit.stage",
    "loop": "basersi.loop.stage",
    "judge": "basersi.judge.stage",
    "procedure_edit": "basersi.procedure_edit.stage",
    "export": "basersi.export.stage",
}
STATE_FILE = "pipeline_state.json"
GATE_HELP = {
    "procedure_edit": ("the View Edit was not accepted (judge.view_edit.accepted) or no entry was labelled ID_SPECIFIC alone (judge.procedure_entry is null; "
                       "see verdicts.txt in the judge out_dir for the labels: GENERAL = removing the entry costs significantly on the benchmarks it was not written for, ID_SPECIFIC = significant only pooled over all benchmarks (in-distribution), UNPROVEN = removing it costs nothing measurable). To go on anyway, "
                       "either set procedure_edit.teacher_traces plus judge.procedure_edit_source_benchmark and judge.procedure_edit_entry_index, or choose a group with "
                       "judge.procedure_entry_group and re-run from judge; --no-gates skips this check"),
    "export": "procedure_edit wrote no spec (procedure_edit.spec); run procedure_edit first or set export.spec",
    "line_search_failed": ("the line search of the Procedure Edit found no step with a finite held-out divergence "
                           "(procedure_edit.line_search_failed gives the reason and the replay folder): there is no spec to export; "
                           "no setting turns this into a pass"),
    "view_edit": ("the view entry was not accepted (judge.floor_accepted is false: the image-token floor of the loop's system does not pass the acceptance test "
                  "against the stock model, see section 2a of verdicts.txt in the judge out_dir; or no judge step with the entries part ran before view_edit). "
                  "The View Edit writes an accepted view entry into the patch "
                  "embedding, so there is nothing to write; --no-gates skips this check"),
    "not_judged": ("the Procedure Edit candidate was NOT JUDGED (judge.procedure_edit.judged is false): it generates more tokens per row than the View Edit system's "
                   "reference frontier covers, so no comparison at equal generated tokens exists and there is neither a verdict nor a tip file; "
                   "judge.procedure_edit.reason, candidate_tokens and frontier_max_tokens give the figures; no setting turns this into a pass"),
}
GATES = {
    "view_edit": ((("judge", "floor_accepted"),),),
    "procedure_edit": ((("judge", "view_edit", "accepted"),), (("judge", "procedure_entry"), ("config", "procedure_edit", "teacher_traces"))),
    "export": ((("procedure_edit", "spec"), ("config", "export", "spec")),),
}


PATH_KEYS = {
    None: ("work_dir", "data_dir", "model"),
    "data": ("development_rows", "sources", "images_dir"),
    "view_edit": ("checkpoint",),
    "loop": ("run_dir", "queue_dir", "start_tip", "start_sources", "start_verdicts", "judge_exports_dir"),
    "judge": ("export_dir", "out_dir", "config_override", "view_edit_checkpoint", "procedure_edit_checkpoint"),
    "procedure_edit": ("teacher_traces", "entry", "work_dir"),
    "export": ("base", "spec", "checkpoint"),
}
REQUIRED = ("work_dir", "data_dir", "model", "base_model")
MODEL_STAGES = ("view_edit", "loop", "procedure_edit")
FINGERPRINT_STAGES = ("view_edit", "loop", "procedure_edit", "export")
EXPORT_SETTINGS = "judge_export"
NOT_FINGERPRINTED = {"loop": ("generations", "stop_before", "judge_exports_dir", EXPORT_SETTINGS)}
OPERATIONAL = {"loop": ("wait_seconds", "poll_seconds"), "judge": ("candidate_read_port", "candidate_read_cards"),
               "procedure_edit": ("work_dir", "memory_fraction", ("fit", "n_threads"), ("serving", "cards"))}
DEVICE_KEYS = {"procedure_edit": ("device",)}
VOLATILE_OUTPUTS = {"data": ("built", "excluded_source"), "loop": (("origin", "code"), "generations_run", "accepted", "reports", "tip_at_start")}
ENGINE_READ_STAGES = ("loop", "judge", "procedure_edit")
INPUT_STAGES = {"data": (), "view_edit": ("judge",), "loop": ("data",), "judge": ("data", "view_edit", "loop", "export", "judge"),
                "procedure_edit": ("data", "judge"), "export": ("view_edit", "procedure_edit")}
BASE_MODEL_STAGES = ("view_edit", "loop", "procedure_edit", "export")
ROWS_STAGES = ("data", "loop", "judge", "procedure_edit")
JUDGE_PARTS = ("entries", "view_edit", "procedure_edit")
CANDIDATE_PARTS = ("view_edit", "procedure_edit")


class PipelineError(C.Refused, RuntimeError):
    pass


def absolute_paths(cfg, base):
    for section, keys in PATH_KEYS.items():
        d = cfg if section is None else cfg.get(section)
        if not isinstance(d, dict):
            continue
        for k in keys:
            v = d.get(k)
            if isinstance(v, str) and v and not os.path.isabs(v):
                d[k] = os.path.normpath(os.path.join(base, v))
    return cfg


MERGED_SECTIONS = ("view_edit", "loop", "procedure_edit", "export")


def check_user_keys(user, source=None):
    shipped = C.load("pipeline")
    source = source or "the pipeline config"
    declared = dict(shipped, **{k: shipped.get(k) or {} for k in MERGED_SECTIONS})
    C.check_keys({k: v for k, v in user.items() if not (k in MERGED_SECTIONS and isinstance(v, dict))}, declared, "pipeline", source)
    for k in MERGED_SECTIONS:
        if isinstance(user.get(k), dict):
            C.check_keys(user[k], C.merge(C.load(k), shipped.get(k) or {}), k, source, what=f"the pipeline.yaml {k} section or {k}.yaml", prefix=[k])
    co = (user.get("judge") or {}).get("config_override")
    if isinstance(co, dict):
        C.check_keys(co, C.load("judge"), "judge", source, prefix=["judge", "config_override"])


def load_config(override=None):
    shipped = absolute_paths(C.load("pipeline"), os.getcwd())
    if override is None:
        cfg = shipped
    else:
        user = C._read(override, "the pipeline config") if isinstance(override, str) else copy.deepcopy(override)
        check_user_keys(user, override if isinstance(override, str) else None)
        section = user.get("procedure_edit")
        if isinstance(section, dict) and isinstance(section.get("strength"), dict):
            C.check_strength(C.load("procedure_edit", {"strength": section["strength"]}))
        base = os.path.dirname(os.path.abspath(override)) if isinstance(override, str) else os.getcwd()
        cfg = C.merge(shipped, absolute_paths(user, base))
        if user.get("base_model") is not None:
            cfg["base_model"] = merged_base_model(shipped["base_model"], user["base_model"])
    missing = [k for k in REQUIRED if not cfg.get(k)]
    if missing:
        raise PipelineError(f"the pipeline config must set {missing}")
    return cfg


def base_model_of(override=None):
    shipped = C.load("pipeline")["base_model"]
    if override is None:
        return shipped
    user = C._read(override, "the base-model config") if isinstance(override, str) else override
    if not isinstance(user.get("base_model"), (dict, type(None))):
        raise C.ConfigError(f"the base-model config {override} must hold a base_model mapping (id, revision, config_sha256, weights)")
    return merged_base_model(shipped, user["base_model"]) if user.get("base_model") is not None else shipped


def merged_base_model(shipped, given):
    if isinstance(given, dict) and "id" not in given:
        return C.merge(shipped, given)
    return copy.deepcopy(given)


def steps(cfg):
    out = []
    for i, item in enumerate(cfg["stages"]):
        if isinstance(item, str):
            item = {"stage": item}
        if not isinstance(item, dict) or "stage" not in item:
            raise C.ConfigError(f"stages[{i}] must be a stage name or a mapping with the keys stage and parts, not {item!r}")
        extra = [k for k in item if k not in ("stage", "parts")]
        if extra:
            raise C.ConfigError(f"stages[{i}] has keys that a step does not declare: " + ", ".join(f"{k}{C.hint(k, ['stage', 'parts'])}" for k in extra)
                                + "; valid keys there: stage, parts (an unknown key would be silently ignored, so it is refused)")
        name = item["stage"]
        if name not in STAGE_MODULES:
            raise C.ConfigError(f"stages[{i}]: unknown stage {name!r}{C.hint(str(name), list(STAGE_MODULES))}; known {sorted(STAGE_MODULES)}")
        parts = item.get("parts")
        if parts is not None and (name != "judge" or not isinstance(parts, list) or not parts or any(p not in JUDGE_PARTS for p in parts)):
            raise C.ConfigError(f"stages[{i}].parts {parts!r}: only a judge step takes parts, a non-empty list of {', '.join(JUDGE_PARTS)}")
        label = name if not parts else f"{name}[{','.join(parts)}]"
        if any(s["label"] == label for s in out):
            raise C.ConfigError(f"stages[{i}]: the step {label} is listed twice; each step keeps one record in pipeline_state.json, so give each step once")
        out.append(dict(index=i, stage=name, parts=parts, label=label))
    if not out:
        raise C.ConfigError("stages is empty: list the steps to run (the shipped pipeline.yaml lists all of them)")
    return out


def loop_run_dir(cfg, mod=None):
    if mod is not None and hasattr(mod, "run_dir_of"):
        return mod.run_dir_of(cfg)
    lc = cfg.get("loop") or {}
    return lc.get("run_dir") or os.path.join(cfg["work_dir"], "loop", lc.get("run_name") or "main")


def kept_state(cfg, step, mod):
    if step["stage"] == "loop":
        d = loop_run_dir(cfg, mod)
        if os.path.exists(os.path.join(d, "state.json")):
            return d
    return None


def state_path(cfg):
    return os.path.join(cfg["work_dir"], STATE_FILE)


def load_state(cfg):
    p = state_path(cfg)
    if os.path.exists(p):
        state = json.load(open(p, encoding="utf-8"))
        if "steps" not in state:
            raise PipelineError(f"{p} holds no per-step fingerprints, so this pipeline cannot tell what its steps were run with; use a new work_dir")
        return state
    return dict(outputs={}, runs=[], steps={}, step_order=[])


def save_state(cfg, state):
    os.makedirs(cfg["work_dir"], exist_ok=True)
    p = state_path(cfg)
    json.dump(state, open(p + ".tmp", "w", encoding="utf-8"), indent=1, default=str)
    os.replace(p + ".tmp", p)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def merged_outputs(state, before=None):
    out = {}
    for label in sorted(state["step_order"], key=lambda k: state["steps"].get(k, {}).get("index", -1)):
        rec = state["steps"].get(label)
        if before is not None and rec and rec.get("index", -1) >= before:
            continue
        if rec and rec.get("outputs") is not None:
            out.setdefault(rec["stage"], {}).update(rec["outputs"])
    return out


def dependents(state, step):
    changed, out = {step["stage"]}, []
    for label, rec in sorted(state["steps"].items(), key=lambda kv: kv[1].get("index", -1)):
        if rec.get("index", -1) <= step["index"] or not changed & set(INPUT_STAGES.get(rec.get("stage"), ())):
            continue
        changed.add(rec["stage"])
        out.append(label)
    return out


def hold_later(state, step):
    for label in dependents(state, step):
        rec = state["steps"][label]
        if rec.get("outputs") is not None and "held_outputs" not in rec:
            rec["held_outputs"], rec["outputs"] = rec["outputs"], None
            rec["held_by"] = dict(step=step["label"], at=now())


def release_later(state, step, changed):
    for label in dependents(state, step):
        rec = state["steps"][label]
        if "held_outputs" not in rec:
            continue
        held = rec.pop("held_outputs")
        rec.pop("held_by", None)
        if changed:
            rec["invalidated_by"] = dict(step=step["label"], at=now())
        else:
            rec["outputs"] = held


def cleared_inputs(state, step):
    out = []
    for label, rec in sorted(state["steps"].items(), key=lambda kv: kv[1].get("index", -1)):
        if rec.get("index", -1) < step["index"] and rec.get("stage") in INPUT_STAGES.get(step["stage"], ()) and rec.get("outputs") is None:
            if rec.get("invalidated_by"):
                out.append((label, f"cleared when {rec['invalidated_by']['step']} changed its outputs"))
            elif rec.get("held_by"):
                out.append((label, f"held since {rec['held_by']['step']} stopped before finishing"))
    return out


def lasting(stage, outputs):
    if not isinstance(outputs, dict):
        return outputs
    out = copy.deepcopy(outputs)
    for key in VOLATILE_OUTPUTS.get(stage, ()):
        path = key if isinstance(key, tuple) else (key,)
        node = out
        for x in path[:-1]:
            node = node.get(x) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return out


def inputs_of(state, step):
    out = {}
    for label, rec in state["steps"].items():
        if rec.get("index", -1) < step["index"] and rec.get("stage") in INPUT_STAGES.get(step["stage"], ()) and rec.get("outputs") is not None:
            out[label] = lasting(rec.get("stage"), rec["outputs"])
    return out


def device_type(value):
    return None if value is None else str(value).split(":")[0]


def operational_free(stage, d):
    out = copy.deepcopy(d) if isinstance(d, dict) else {}
    for key in OPERATIONAL.get(stage, ()):
        path = key if isinstance(key, tuple) else (key,)
        node = out
        for x in path[:-1]:
            node = node.get(x) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(path[-1], None)
    for key in DEVICE_KEYS.get(stage, ()):
        if key in out:
            out[key] = device_type(out[key])
    return out


def without(d, keys):
    return {k: v for k, v in (d or {}).items() if k not in keys}


def stage_settings(cfg, stage):
    if stage == "data":
        return dict(benchmarks=C.load("benchmarks"), sources=C.load("sources"))
    if stage == "loop":
        from basersi.judge import judge as JV
        return dict(loop=operational_free("loop", without(C.load("loop"), (EXPORT_SETTINGS,))), judge=JV.settings_of_pipeline(cfg))
    if stage == "judge":
        from basersi.judge import judge as JV
        override = (cfg.get("judge") or {}).get("config_override")
        return dict(judge=operational_free("judge", JV.judge_config(override)), tail_chars=JV.tail_chars(),
                    export_settings=C.merge(C.load("loop")[EXPORT_SETTINGS], ((cfg.get("loop") or {}).get(EXPORT_SETTINGS) or {})))
    return operational_free(stage, C.load(stage))


def fingerprint_material(cfg, state, step):
    stage = step["stage"]
    section = without(cfg.get(stage), NOT_FINGERPRINTED.get(stage, ()))
    if stage in MERGED_SECTIONS:
        section = C.merge(without(C.load(stage), NOT_FINGERPRINTED.get(stage, ())), section)
    section = operational_free(stage, section)
    if step["parts"]:
        section["parts"] = list(step["parts"])
    material = dict(stage=stage, section=section, config=stage_settings(cfg, stage), data_dir=cfg.get("data_dir"), work_dir=cfg.get("work_dir"),
                    inputs=inputs_of(state, step), code=C.code_identity())
    if stage in ROWS_STAGES:
        material["rows_manifest_sha256"] = C.rows_manifest_sha256()
    if stage in BASE_MODEL_STAGES:
        material.update(model=cfg.get("model"), base_model=without(cfg.get("base_model"), ("weights_check",)))
    if stage in ENGINE_READ_STAGES:
        material["read_settings_sha256"] = C.read_settings_sha256(direct=stage != "loop")
    return FP.canonical(material)


def fingerprint(cfg, state, step):
    return FP.digest(fingerprint_material(cfg, state, step))


def changed_parts(old, new):
    if not isinstance(old, dict):
        return ["the whole record (the step's record holds no fingerprint material)"]
    return [k for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]


def run_step(cfg, state, step):
    label = step["label"]
    material = fingerprint_material(cfg, state, step)
    fp = FP.digest(material)
    before = state["steps"].get(label)
    changed = before is not None and before.get("fingerprint") is not None and before.get("fingerprint") != fp
    other_code = None
    if changed and accepts_other_code(cfg) and changed_parts(before.get("material"), material) == ["code"]:
        other_code = dict(recorded=before["material"]["code"], now=material["code"], key="resume.accept_outputs_of_other_code")
        print(f"[basersi] {label}: its outputs were made by other code ({other_code['recorded']}; now {other_code['now']}); reused because "
              f"resume.accept_outputs_of_other_code is true", file=sys.stderr, flush=True)
        changed = False
    mod = importlib.import_module(STAGE_MODULES[step["stage"]])
    if changed and step["stage"] in FINGERPRINT_STAGES and not getattr(mod, "HONOURS_FINGERPRINT", False):
        kept = kept_state(cfg, step, mod)
        if kept:
            raise PipelineError(f"step {label} ran before in {kept} with another config, other inputs or other code (changed: {', '.join(changed_parts(before.get('material'), material))});"
                                f" it would continue that state. Move it away and start afresh:\n  {FP.move_commands([kept])}\n"
                                f"or give this run its own folder ({step['stage']}.run_name or {step['stage']}.run_dir), or a new work_dir")
    if label not in state["step_order"]:
        state["step_order"].append(label)
    previous = (before or {}).get("outputs")
    if previous is None:
        previous = (before or {}).get("previous_outputs")
    state["steps"][label] = dict(stage=step["stage"], index=step["index"], fingerprint=fp, material=material, outputs=None, previous_outputs=previous, started=now())
    if other_code:
        state["steps"][label]["outputs_of_other_code"] = other_code
        records = state.setdefault("outputs_of_other_code", [])
        if not any(r.get("step") == label and r.get("recorded") == other_code["recorded"] and r.get("now") == other_code["now"] for r in records):
            records.append(dict(other_code, step=label, at=now()))
    hold_later(state, step)
    state["outputs"] = merged_outputs(state)
    save_state(cfg, state)
    stage_cfg = copy.deepcopy(cfg)
    if step["parts"]:
        stage_cfg[step["stage"]] = dict(stage_cfg.get(step["stage"]) or {}, parts=list(step["parts"]))
    stage_cfg["outputs"] = copy.deepcopy(merged_outputs(state, before=step["index"]))
    stage_cfg["fingerprint"] = dict(value=fp, changed=changed, outputs_of_other_code=other_code)
    stage_cfg[FP.OTHER_CODE_KEY] = accepts_other_code(cfg)
    rec = dict(step=label, index=step["index"], started=now(), code=material["code"], fingerprint=fp)
    if other_code:
        rec["outputs_of_other_code"] = other_code
    result = mod.run(stage_cfg)
    rec["finished"] = now()
    state["steps"][label].update(outputs=result or {}, finished=rec["finished"])
    state["steps"][label].pop("previous_outputs", None)
    release_later(state, step, previous is None or FP.digest(FP.canonical(lasting(step["stage"], result or {}))) != FP.digest(FP.canonical(lasting(step["stage"], previous))))
    if isinstance(result, dict) and result.get("origin") is not None:
        state["steps"][label]["origin"] = rec["origin"] = result["origin"]
    state["outputs"] = merged_outputs(state)
    rec["outputs"] = result
    state["runs"].append(rec)
    save_state(cfg, state)
    return result


def lookup(cfg, state, path):
    v = cfg if path[0] == "config" else state["outputs"]
    for k in path[1:] if path[0] == "config" else path:
        v = v.get(k) if isinstance(v, dict) else None
    return v


def closed_gate(cfg, state, step):
    for alternatives in GATES.get(step["stage"], ()):
        if not any(lookup(cfg, state, path) for path in alternatives):
            return " or ".join(".".join(path) for path in alternatives)
    return None


def run(cfg, only=None, start=None, stop=None):
    all_steps = steps(cfg)
    labels = [s["label"] for s in all_steps]
    names = [s["stage"] for s in all_steps]

    def find(key):
        if key in labels:
            return labels.index(key)
        if key in names:
            return names.index(key)
        if str(key).isdigit() and int(key) < len(all_steps):
            return int(key)
        raise PipelineError(f"no step {key!r}; steps: {labels}")
    if only is not None:
        chosen = [all_steps[find(only)]]
    else:
        a = find(start) if start is not None else 0
        b = find(stop) if stop is not None else len(all_steps) - 1
        chosen = all_steps[a:b + 1]
    state = load_state(cfg)
    state["config"] = cfg
    state.pop("stopped", None)
    if any(uses_model(cfg, st) for st in chosen):
        from basersi.model_check import check_base_model
        state["base_model_check"] = check_base_model(cfg["model"], cfg["base_model"])
    results = {}
    for st in chosen:
        cleared = cleared_inputs(state, st)
        if cleared:
            raise PipelineError(f"step {st['label']} would read the outputs of " + "; ".join(f"{lab} ({why})" for lab, why in cleared)
                                + f": such outputs are not used until they are made again from the current inputs; run them again first (pipeline --from {cleared[0][0]})")
        gate = closed_gate(cfg, state, st) if cfg.get("acceptance_gates", True) else None
        if gate:
            state["stopped"] = dict(step=st["label"], gate=gate, why=GATE_HELP.get(st["stage"]), at=now())
            save_state(cfg, state)
            results["stopped"] = state["stopped"]
            break
        results[st["label"]] = run_step(cfg, state, st)
        held = outcome_stop(st["stage"], results[st["label"]])
        if held:
            gate, why, figures = held
            state["stopped"] = dict(step=st["label"], gate=gate, why=why, at=now(), **figures)
            save_state(cfg, state)
            results["stopped"] = state["stopped"]
            break
    return results


def accepts_other_code(cfg):
    return bool((cfg.get("resume") or {}).get("accept_outputs_of_other_code"))


def uses_model(cfg, step):
    if step["stage"] == "judge":
        return bool((cfg.get("judge") or {}).get("read_candidates", True)) and any(p in (step["parts"] or JUDGE_PARTS) for p in CANDIDATE_PARTS)
    return step["stage"] in MODEL_STAGES


def not_judged(result):
    pe = (result or {}).get("procedure_edit") if isinstance(result, dict) else None
    return "judge.procedure_edit.judged" if isinstance(pe, dict) and pe.get("judged") is False else None


def outcome_stop(stage, result):
    if stage == "judge" and not_judged(result):
        return "judge.procedure_edit.judged", GATE_HELP["not_judged"], {k: result["procedure_edit"].get(k) for k in ("reason", "candidate_tokens", "frontier_max_tokens")}
    if stage == "procedure_edit" and isinstance(result, dict) and result.get("line_search_failed"):
        return "procedure_edit.line_search_failed", GATE_HELP["line_search_failed"], dict(line_search_failed=result["line_search_failed"])
    return None
