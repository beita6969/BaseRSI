import argparse
import json

from basersi import config

HELP = "export: write a Procedure Edit spec into a full copy of a checkpoint and check that nothing outside the spec changed"
BASE_MODEL = ("yaml file whose base_model block (id, revision, config_sha256, weights) replaces the one in basersi/config/pipeline.yaml; it names the stock checkpoint; "
              "a View Edit base is traced to its stock folder through the basersi_edit.json it carries")
RECORD = ("the edited folder gets a basersi_edit.json naming every edit applied to it and a content_sha256 that changes with them; "
          "an edit in place first marks the folder as pending, so a stopped edit is never taken for the old or the new checkpoint")


REMAP_EXPECTED = "a JSON object {array file named in the spec: array file on this machine}"


def load_remap(path):
    from basersi import fingerprint as fp
    if not path:
        return None
    return fp.given_json(path, "--remap", REMAP_EXPECTED, lambda v: isinstance(v, dict) and all(isinstance(x, str) for x in v.values()))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="basersi export", description=HELP, epilog=RECORD + "; " + config.EXIT_CODES)
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("checkpoint", epilog=config.EXIT_CODES, formatter_class=argparse.RawDescriptionHelpFormatter, help="copy the base checkpoint (links resolved) and apply the spec to the copy")
    e.add_argument("--base", required=True, help="checkpoint to edit (the View Edit checkpoint for the final model); an edited base is accepted here: "
                                                  "the base-model check follows its basersi_edit.json back to the stock folder and checks that")
    e.add_argument("--spec", required=True, help="spec json written by the Procedure Edit")
    e.add_argument("--out", required=True, help="checkpoint folder; the copy is made in OUT.partial and renamed when checked, with its report in OUT.json; "
                                                 "an OUT made with the same inputs is reused, one made with other inputs is refused; "
                                                 "a OUT.partial left by a stopped run is moved aside to OUT.partial.set-aside-N, never reused")
    e.add_argument("--remap", default=None, help="json {array file named in the spec: array file on this machine}; relative names are read next to the spec")
    e.add_argument("--base-model-config", default=None, help=BASE_MODEL)
    e.add_argument("--accept-outputs-of-other-code", action="store_true", help="accept and reuse outputs whose recorded inputs differ from this run only in the package code (the pipeline key resume.accept_outputs_of_other_code); "
                        "each reuse is printed and recorded; any other difference is still refused")
    p = sub.add_parser("patch", epilog=config.EXIT_CODES, formatter_class=argparse.RawDescriptionHelpFormatter, help="apply a spec in place to an existing full copy (optionally restoring the rows of a previous spec first)")
    p.add_argument("--folder", required=True, help="full copy of --reference to edit in place; if it already holds an edit (basersi_edit.json), "
                                                    "it is written only when --previous-spec is the spec recorded there, otherwise nothing is written")
    p.add_argument("--reference", required=True, help="checkpoint the copy was made from (stock or View Edit; checked like --base)")
    p.add_argument("--spec", required=True, help="spec json to apply")
    p.add_argument("--previous-spec", default=None, help="spec applied to the copy before; its rows are restored from the reference first")
    p.add_argument("--remap", default=None, help="json {array file named in the spec: array file on this machine}; relative names are read next to the spec")
    p.add_argument("--base-model-config", default=None, help=BASE_MODEL)
    a = ap.parse_args(argv)
    from basersi import fingerprint as fp
    from basersi.export import stage
    from basersi.export.patch import patch_folder
    from basersi.export.stage import check_export_base
    from basersi.pipeline import base_model_of
    from basersi.export.patch import load_spec
    for arg, folder in (("--base", getattr(a, "base", None)), ("--folder", getattr(a, "folder", None)), ("--reference", getattr(a, "reference", None))):
        if folder is not None:
            fp.given_checkpoint(folder, arg)
    load_spec(a.spec, "--spec")
    if a.base_model_config:
        fp.given_path(a.base_model_config, "--base-model-config")
    base_model = base_model_of(a.base_model_config)
    remap = load_remap(a.remap)
    if a.command == "checkpoint":
        report = stage.run({"base_model": base_model, "export": dict(base=a.base, spec=a.spec, checkpoint=a.out, remap=remap),
                            fp.OTHER_CODE_KEY: a.accept_outputs_of_other_code})
        report = dict(report, **fp.reuse_records_of(report["checkpoint"] + ".json"))
    else:
        checked = check_export_base(a.reference, base_model)
        report = patch_folder(a.folder, a.reference, a.spec, previous_spec_path=a.previous_spec, remap=remap)
        report["base_check"] = checked
    print(json.dumps(report, indent=1))
    return 0
