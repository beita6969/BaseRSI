import os

from basersi import config as C
from basersi import fingerprint as fp
from basersi.view_edit.build import build_view_checkpoint, view_inputs

HONOURS_FINGERPRINT = True

SETTINGS = ("patch_size", "min_image_tokens", "max_image_tokens")
KEY = "view_edit.checkpoint"


def broken_links(folder):
    return sorted(name for name in os.listdir(folder) if os.path.islink(os.path.join(folder, name)) and not os.path.exists(os.path.join(folder, name)))


def checked_folder(out, report_path, recorded_fingerprint):
    edit = fp.last_edit(out, "view_edit")
    if edit is None or edit["inputs_fingerprint"] != recorded_fingerprint:
        raise fp.StaleOutput(f"{out} does not hold the View Edit its report {report_path} describes (its {fp.MARKER} differs or is missing): "
                             f"{fp.way_forward([out, report_path], KEY)}")
    written = edit["report"].get("written_sha256")
    changed = sorted(n for n, sha in (written or {}).items()
                     if os.path.islink(os.path.join(out, n)) or not os.path.isfile(os.path.join(out, n)) or fp.file_sha256(os.path.join(out, n)) != sha)
    if not written or changed:
        what = f"{len(changed)} of the files the build wrote changed after it ({', '.join(changed[:3])})" if written else "it has no record of the files the build wrote"
        raise fp.StaleOutput(f"{out}: {what}; build the View Edit again: {fp.way_forward([out, report_path], KEY)}")
    broken = broken_links(out)
    if broken:
        raise fp.StaleOutput(f"{out} links into the stock folder, and {len(broken)} links no longer resolve (e.g. {broken[0]}): the stock folder moved; "
                             f"build the View Edit again: {fp.way_forward([out, report_path], KEY)}")
    return edit


def run(config):
    with fp.other_code(config.get(fp.OTHER_CODE_KEY)):
        return build_or_reuse(config)


def build_or_reuse(config):
    cfg = C.load("view_edit", config.get("view_edit") or {})
    if not os.path.isdir(config["model"]):
        raise C.MissingInput(f"the View Edit model folder {config['model']} does not exist")
    if config.get("base_model"):
        from basersi.model_check import check_base_model
        check_base_model(config["model"], config["base_model"])
    out = os.path.abspath(cfg.get("checkpoint") or os.path.join(config["work_dir"], "view_edit", "checkpoint")).rstrip("/")
    report_path = out + ".json"
    inputs = view_inputs(config["model"], *(cfg[k] for k in SETTINGS))
    if os.path.exists(report_path):
        report = fp.read_json(report_path)
        matched = fp.match(report.get("inputs_fingerprint"), report.get("inputs"), inputs, out, reads_rows=False)
        if not matched:
            raise fp.StaleOutput(fp.stale_message(report_path, dict(inputs=report.get("inputs")) if report.get("inputs") else None, inputs,
                                                  records=[out], key=KEY, reads_rows=False))
        if not os.path.isdir(out):
            raise fp.StaleOutput(f"{report_path} describes {out}, which is gone: {fp.way_forward([report_path], KEY)}")
        checked_folder(out, report_path, report["inputs_fingerprint"])
        updated = fp.with_other_code_record(report, matched)
        if updated is not report:
            fp.write_json_atomic(report_path, updated, indent=1)
        return fp.without_reuse_record(updated)
    if os.path.lexists(out):
        edit = fp.last_edit(out, "view_edit")
        matched = fp.match(edit["inputs_fingerprint"], edit["report"].get("inputs"), inputs, out, reads_rows=False) if edit else None
        if not matched:
            what = "a View Edit made with other inputs" if edit else f"no View Edit record ({fp.MARKER})"
            raise fp.StaleOutput(f"{out} exists without its report {report_path} and holds {what}: {fp.way_forward([out], KEY)}")
        checked_folder(out, report_path, edit["inputs_fingerprint"])
        report = fp.with_other_code_record(dict(edit["report"], recovered_from=os.path.join(out, fp.MARKER)), matched)
    else:
        report = build_view_checkpoint(config["model"], out, *(cfg[k] for k in SETTINGS))
    fp.write_json_atomic(report_path, report, indent=1)
    return fp.without_reuse_record(report)
