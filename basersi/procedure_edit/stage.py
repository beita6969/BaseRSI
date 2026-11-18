import os

from basersi import config as C
from basersi import fingerprint as fp
from basersi.procedure_edit.rounds import Compiler, check_names

HONOURS_FINGERPRINT = True


def settings(config):
    cfg = C.load("procedure_edit", config.get("procedure_edit") or {})
    check_names(cfg)
    return cfg


def run(config):
    with fp.other_code(config.get(fp.OTHER_CODE_KEY)):
        return compile_or_reuse(config)


def compile_or_reuse(config):
    cfg = settings(config)
    outputs = config.get("outputs") or {}
    entry = (outputs.get("judge") or {}).get("procedure_entry") or cfg.get("entry")
    if isinstance(entry, str):
        entry = fp.given_json(entry, "procedure_edit.entry (--entry)", "a JSON object: a procedure entry or a judge result holding procedure_entry",
                              lambda v: isinstance(v, dict))
    if isinstance(entry, dict) and isinstance(entry.get("procedure_entry"), dict):
        entry = entry["procedure_entry"]
    if not cfg.get("teacher_traces") and not entry:
        raise C.MissingInput("procedure_edit needs outputs['judge']['procedure_entry'], procedure_edit.entry or procedure_edit.teacher_traces")
    rows_path = (outputs.get("data") or {}).get("development_rows") or os.path.join(config["data_dir"], C.load("benchmarks")["development"]["file"])
    missing = [f"{what} {path}" for what, path, exists in (("model folder", config["model"], os.path.isdir), ("development rows", rows_path, os.path.isfile),
                                                            ("teacher traces", cfg.get("teacher_traces"), os.path.isfile)) if path and not exists(path)]
    if missing:
        raise C.MissingInput(f"the Procedure Edit input does not exist: {'; '.join(missing)}")
    if config.get("base_model"):
        from basersi.model_check import check_base_model
        check_base_model(config["model"], config["base_model"])
    from basersi.engine.reads import read_rows
    compiler = Compiler(cfg, C.benchmarks(), config["model"], cfg.get("work_dir") or os.path.join(config["work_dir"], "procedure_edit"),
                        rows_path, read_rows, entry=entry, teacher_traces=cfg.get("teacher_traces"))
    result = compiler.compile()
    result["entry_id"] = None if cfg.get("teacher_traces") else entry["cid"]
    result["teacher_traces_given"] = bool(cfg.get("teacher_traces"))
    return result
