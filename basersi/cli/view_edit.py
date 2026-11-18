import argparse
import json

from basersi import config

CHECK_EXIT = "\n  view-edit check: 1 when the checkpoint differs from the stock one outside the View Edit"
HELP = "View Edit: resample the vision patch kernel to a finer patch and set the image-token bounds of the processor"


def main(argv=None):
    settings = config.load("view_edit")
    ap = argparse.ArgumentParser(prog="basersi view-edit", description=HELP, epilog=config.EXIT_CODES + CHECK_EXIT, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", epilog=config.EXIT_CODES, formatter_class=argparse.RawDescriptionHelpFormatter, help="write the View Edit checkpoint (patched kernel shard, rewritten configs, absolute links to every other file of the stock folder, which must stay in place)")
    b.add_argument("--model", required=True, help="stock checkpoint folder; it is checked against base_model, and an edited checkpoint (one holding basersi_edit.json) is refused")
    b.add_argument("--out", required=True, help="checkpoint folder; it is built in OUT.partial, checked, given a basersi_edit.json record and then renamed, "
                                                 "with its report in OUT.json; an OUT made with the same inputs is reused, one made with other inputs is refused; "
                                                 "a OUT.partial left by a stopped run is moved aside to OUT.partial.set-aside-N, never reused")
    b.add_argument("--base-model-config", default=None, help="yaml file whose base_model block (id, revision, config_sha256, weights) replaces "
                                                             "the one in basersi/config/pipeline.yaml; it names the checkpoint --model must hold")
    b.add_argument("--patch-size", type=int, default=settings["patch_size"], help="new vision patch size (must divide the stock patch size)")
    b.add_argument("--min-image-tokens", type=int, default=settings["min_image_tokens"], help="smallest number of merged tokens per image (0 keeps the stock lower bound)")
    b.add_argument("--max-image-tokens", type=int, default=settings["max_image_tokens"], help="largest number of merged tokens per image")
    b.add_argument("--accept-outputs-of-other-code", action="store_true", help="accept and reuse outputs whose recorded inputs differ from this run only in the package code (the pipeline key resume.accept_outputs_of_other_code); "
                        "each reuse is printed and recorded; any other difference is still refused")
    c = sub.add_parser("check", epilog=config.EXIT_CODES + CHECK_EXIT, formatter_class=argparse.RawDescriptionHelpFormatter, help="check that a View Edit checkpoint differs from the stock one only in the patch kernel and the patch / bounds entries of the configs")
    c.add_argument("--model", required=True, help="stock checkpoint folder")
    c.add_argument("--out", required=True, help="View Edit checkpoint folder")
    a = ap.parse_args(argv)
    from basersi import fingerprint as fp
    from basersi.view_edit.check import check_view_checkpoint
    fp.given_checkpoint(a.model, "--model")
    if a.command == "build":
        if a.base_model_config:
            fp.given_path(a.base_model_config, "--base-model-config")
        from basersi.model_check import check_base_model
        from basersi.pipeline import base_model_of
        from basersi.view_edit import stage
        base_model = base_model_of(a.base_model_config)
        checked = check_base_model(a.model, base_model)
        settings = dict(patch_size=a.patch_size, min_image_tokens=a.min_image_tokens, max_image_tokens=a.max_image_tokens, checkpoint=a.out)
        result = dict(stage.run({"model": a.model, "base_model": base_model, "view_edit": settings, fp.OTHER_CODE_KEY: a.accept_outputs_of_other_code}), base_check=checked)
        result.update(fp.reuse_records_of(result["checkpoint"] + ".json"))
    else:
        fp.given_checkpoint(a.out, "--out")
        result = check_view_checkpoint(a.model, a.out)
    print(json.dumps(result, indent=1))
    return 0 if (result.get("check") or result)["result"] == "ok" else 1
