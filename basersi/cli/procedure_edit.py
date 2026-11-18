import argparse
import json
import os

from basersi import config

HELP = "Procedure Edit: compile a routed procedure into gated MLP units over several teacher / student rounds and write the spec"
STRENGTH = ("the step of the final spec. By default (procedure_edit.strength.step: null) a line search replays every step of "
            "procedure_edit.strength.steps on the last round's own held-out reads of the selection fold (the last student, or the unedited model when "
            "rounds is 2) and takes the step with the lowest mean divergence from the teacher (the smaller step on a tie); a step given here is used as "
            "it is, without the search; the line search of the published model selects 1.8. The student of every earlier round is exported at the step that the same line search selects for its round on the held-out reads of the current student (procedure_edit.student.line_search; false keeps the step of procedure_edit.student.cell)")
NOTES = ("Before any fit, the site of the entry is measured on the unedited model's own reads of the routed questions of the selection fold: the share of the divergence from "
         "the teacher that is removed when the outputs of the linear-attention layers are set to the teacher's (procedure_edit.site; records in "
         "site/own, the mean share in site/linear_fraction.json and in the printed result). "
         "The key slope and threshold (procedure_edit.key) were set on the reads of the selection fold "
         "(select_fold), which are also the reads the line search and the published teacher-forced numbers are measured on. The list rules (seeds, totals, "
         "key-row exclusions in procedure_edit.lists) are the constants of the published build; lists are drawn from the teacher traces at hand, so a run "
         "from new teacher reads follows the same rules and gives other lists and captures. "
         "Position keys keep the procedure text of routed rows (position_keys.drop_prefix: false) while the route key and the gap captures drop it "
         "(route_key.drop_prefix: true), as in the published build. "
         "A step stopped part-way is resumed on the next run: finished files are reused only when their recorded inputs (settings, input files, "
         "engine read settings and the package code) are unchanged, and a half-built student copy (students/roundN.partial) is moved aside to "
         "roundN.partial.set-aside-K and built again.")
ENTRY_EXPECTED = "a JSON object: a procedure entry or a judge result holding procedure_entry"
BASE_MODEL = "yaml file whose base_model block (id, revision, config_sha256, weights) replaces the one in basersi/config/pipeline.yaml; it names the checkpoint --model must hold"


def base_model(path):
    from basersi.pipeline import base_model_of
    return base_model_of(path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="basersi procedure-edit", description=HELP, epilog="--step: " + STRENGTH + ". " + NOTES + " " + config.EXIT_CODES + "; procedure-edit returns 3 when no step of the line search has a finite divergence (line_search_failed in the printed json, spec null)")
    ap.add_argument("--model", required=True, help="stock checkpoint folder (the edit is fitted on the stock weights and stacked on the View Edit only at export); it is checked against base_model, and an edited checkpoint (one holding basersi_edit.json) is refused")
    ap.add_argument("--work-dir", required=True, help="run folder; every step writes its files there with a record of its inputs, reuses them only when the inputs are the same, and stops with an error naming the file when they differ")
    ap.add_argument("--rows", required=True, help="development rows file")
    ap.add_argument("--entry", default=None, help="json of the procedure entry (procedure, routed_signal_rows, view_settings, cid), or a judge result holding one (procedure_entry)")
    ap.add_argument("--teacher-traces", default=None, help="existing teacher traces file; skips the teacher reads. Every row id must be a development signal row (show / monitor) of the configured benchmarks in --rows; roles and row content are taken from --rows, never from this file, and a validation row, an unknown id or a row of another benchmark is refused")
    ap.add_argument("--config", default=None, help="yaml file deep-merged over basersi/config/procedure_edit.yaml")
    ap.add_argument("--step", type=float, default=None, help=STRENGTH)
    ap.add_argument("--base-model-config", default=None, help=BASE_MODEL)
    ap.add_argument("--accept-outputs-of-other-code", action="store_true", help="accept and reuse outputs whose recorded inputs differ from this run only in the package code (the pipeline key resume.accept_outputs_of_other_code); "
                        "each reuse is printed and recorded; any other difference is still refused")
    a = ap.parse_args(argv)
    from basersi import fingerprint as fp
    from basersi.procedure_edit import stage
    fp.given_checkpoint(a.model, "--model")
    fp.given_path(a.rows, "--rows")
    for arg, path in (("--teacher-traces", a.teacher_traces), ("--config", a.config), ("--base-model-config", a.base_model_config)):
        if path:
            fp.given_path(path, arg)
    if a.entry:
        fp.given_json(a.entry, "--entry", ENTRY_EXPECTED, lambda v: isinstance(v, dict))
    override = config.load("procedure_edit", a.config) if a.config else {}
    override = dict(override, work_dir=a.work_dir)
    if a.teacher_traces:
        override["teacher_traces"] = a.teacher_traces
    if a.step is not None:
        override["strength"] = dict(override.get("strength") or {}, step=a.step)
    outputs = dict(data=dict(development_rows=a.rows))
    if a.entry:
        override["entry"] = os.path.abspath(a.entry)
    result = stage.run({"model": a.model, "base_model": base_model(a.base_model_config), "work_dir": a.work_dir, "data_dir": ".", "procedure_edit": override,
                        "outputs": outputs, fp.OTHER_CODE_KEY: a.accept_outputs_of_other_code})
    print(json.dumps(result, indent=1))
    return 3 if result.get("line_search_failed") else 0
