import os

from basersi import config as C
from basersi import fingerprint as fp

HONOURS_FINGERPRINT = True
KEY = "export.checkpoint"


def label(spec):
    from basersi.procedure_edit.spec import describe
    try:
        cell = describe(spec)
        return dict(cell=cell["cell"], gate=cell["gate"], step=cell["step"])
    except ValueError:
        return None


def stock_folder(base):
    edits = fp.marker_edits(base)
    if not edits:
        return base, None
    first = edits[0]
    source = first["report"].get("source") or first["report"].get("reference")
    if not source or not os.path.isdir(source):
        raise fp.StaleOutput(f"{base} was made from {source!r} ({fp.MARKER}), which is gone: the stock folder must stay in place")
    if fp.checkpoint_identity(source) != first["report"]["inputs"]["model" if first["stage"] == "view_edit" else "base"]:
        raise fp.StaleOutput(f"{source} is no longer the folder {base} was made from (its files changed; see {os.path.join(base, fp.MARKER)})")
    return source, first["stage"]


def check_export_base(base, base_model):
    from basersi.model_check import BaseModelError, check_base_model
    source, made_by = stock_folder(base)
    try:
        return dict(check_base_model(source, base_model), checked_folder=source, base_made_by=made_by)
    except BaseModelError as e:
        if made_by is None:
            raise BaseModelError(f"{e} ({base} carries no {fp.MARKER}, so it is taken as the stock folder itself; a View Edit checkpoint built by "
                                 f"basersi names its stock folder there)") from None
        raise


def run(config):
    with fp.other_code(config.get(fp.OTHER_CODE_KEY)):
        return build_or_reuse(config)


def build_or_reuse(config):
    from basersi.export.patch import export_checkpoint, load_spec, spec_inputs
    cfg = C.load("export", config.get("export") or {})
    outputs = config.get("outputs") or {}
    base = cfg.get("base") or (outputs.get("view_edit") or {}).get("checkpoint")
    spec_path = cfg.get("spec") or (outputs.get("procedure_edit") or {}).get("spec")
    failed = (outputs.get("procedure_edit") or {}).get("line_search_failed")
    if not cfg.get("spec") and failed:
        raise C.MissingInput(f"export: the Procedure Edit wrote no spec ({failed.get('reason')}; table {failed.get('table')})")
    if not base or not spec_path:
        raise C.MissingInput("export needs the View Edit checkpoint and the Procedure Edit spec (outputs or export.base / export.spec)")
    missing = [f"{what} {path}" for what, path, exists in (("base folder", base, os.path.isdir), ("spec", spec_path, os.path.isfile)) if not exists(path)]
    if missing:
        raise C.MissingInput(f"the export input does not exist: {'; '.join(missing)}")
    base_check = check_export_base(base, config["base_model"]) if config.get("base_model") else None
    out = os.path.abspath(cfg.get("checkpoint") or os.path.join(config["work_dir"], "export", "checkpoint")).rstrip("/")
    report_path = out + ".json"
    inputs = spec_inputs(base, spec_path, cfg.get("remap"))
    if os.path.exists(report_path):
        report = fp.read_json(report_path)
        matched = fp.match(report.get("inputs_fingerprint"), report.get("inputs"), inputs, out, reads_rows=False)
        if not matched:
            raise fp.StaleOutput(fp.stale_message(report_path, dict(inputs=report.get("inputs")) if report.get("inputs") else None, inputs,
                                                  records=[out], key=KEY, reads_rows=False))
        edit = fp.last_edit(out, "export") if os.path.isdir(out) else None
        if edit is None or edit["inputs_fingerprint"] != report["inputs_fingerprint"]:
            raise fp.StaleOutput(f"{out} does not hold the export its report {report_path} describes (gone, edited in place, or its {fp.MARKER} differs): "
                                 f"{fp.way_forward([out, report_path], KEY)}")
        updated = fp.with_other_code_record(report, matched)
        if updated is not report:
            fp.write_json_atomic(report_path, updated, indent=1)
        return fp.without_reuse_record(updated)
    if os.path.lexists(out):
        edit = fp.last_edit(out, "export")
        matched = fp.match(edit["inputs_fingerprint"], edit["report"].get("inputs"), inputs, out, reads_rows=False) if edit else None
        if not matched:
            what = "an export made with other inputs" if edit else f"no export record ({fp.MARKER})"
            raise fp.StaleOutput(f"{out} exists without its report {report_path} and holds {what}: {fp.way_forward([out], KEY)}")
        report = fp.with_other_code_record(dict(edit["report"], recovered_from=os.path.join(out, fp.MARKER)), matched)
    else:
        report = export_checkpoint(base, spec_path, out, remap=cfg.get("remap"), extra=dict(checkpoint=out, label=label(load_spec(spec_path))), key=KEY)
    report["base_check"] = base_check
    fp.write_json_atomic(report_path, report, indent=1)
    return fp.without_reuse_record(report)
