import argparse
import json

from basersi import config as C
from basersi.judge import stage_entries, stage_procedure_edit, stage_view_edit

HELP = "acceptance tests: the loop tip's entries, the View Edit, the Procedure Edit"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="basersi judge", description=HELP, epilog=C.EXIT_CODES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=["entries", "view-edit", "procedure-edit"], help="which acceptance test to run")
    ap.add_argument("--rows", required=True, help="development rows file (JSON lines whose field role is show / monitor / judge)")
    ap.add_argument("--exports", required=True, help="directory of the read exports named in judge.yaml 'exports'")
    ap.add_argument("--out", required=True, help="output directory (verdict text, json and tip files)")
    ap.add_argument("--config", default=None, help="yaml file merged over basersi/config/judge.yaml")
    ap.add_argument("--part", choices=["signal", "judge", "both"], default="both",
                    help="view-edit / procedure-edit: the signal-row computation, the JUDGE-row verdict (needs the signal json), or both")
    ap.add_argument("--source-benchmark", default=None, help="procedure-edit: source benchmark of the entry the cell was compiled from")
    ap.add_argument("--entry-index", type=int, default=None, help="procedure-edit: index of that entry in the tip's fired table")
    a = ap.parse_args(argv)
    cfg = C.load("judge", a.config)
    if a.source_benchmark is not None:
        cfg["procedure_edit_source_benchmark"] = a.source_benchmark
    if a.entry_index is not None:
        cfg["procedure_edit_entry_index"] = a.entry_index
    if a.test == "procedure-edit" and (cfg.get("procedure_edit_source_benchmark") is None or cfg.get("procedure_edit_entry_index") is None):
        ap.error("procedure-edit needs --source-benchmark and --entry-index (or both keys in --config)")
    if a.test == "entries":
        res = stage_entries.run(a.rows, a.exports, a.out, cfg)
    else:
        mod = stage_view_edit if a.test == "view-edit" else stage_procedure_edit
        ctx = mod.Context(a.rows, a.exports, a.out, cfg)
        res = {}
        if a.part in ("signal", "both"):
            res.update(mod.run_signal(ctx))
        if a.part in ("judge", "both"):
            res.update(mod.run_judge(ctx))
    print(json.dumps(res, indent=1))
    return 0
