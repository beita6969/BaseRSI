import argparse
import json
import textwrap

import yaml

from basersi import config
from basersi import fingerprint as FP
from basersi.loop import stage
from basersi.loop.generation import Refuse, verdicts_from_audit

HELP = "run generations of the harness-search loop, or initialise a run from a start tip"
EPILOG = textwrap.fill("One model does all the work of the loop: the executor, served by 'basersi serve server'. It answers the reads, the route questions and "
          "the proposer's chats (the executor writes the new entries for itself, from cards of its own wrong and right reads); "
          "all requests go through the file queue that 'basersi serve dispatcher' sends to the servers. "
          "A run folder records its origin (queue, rows, loop and engine settings, model) and is continued only with the same origin. "
          "Settings given here override the --config file, which overrides config/loop.yaml.", 110) + "\n\n" + config.EXIT_CODES


def parser():
    p = argparse.ArgumentParser(prog="basersi loop", description=HELP, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generation", help="run one or more generations: signals, candidates, prescreen, judge, update, report", epilog=EPILOG,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    g.add_argument("--run-dir", required=True, help="run directory (state.json, tips/, report_g<NNN>.txt, ledger.jsonl, shrinkage.jsonl)")
    g.add_argument("--rows", required=True, help="development rows file (show / monitor / judge rows), checked against the shipped row manifest")
    g.add_argument("--queue-dir", default=None, help="file-queue directory shared with the dispatcher (q/in, q/out, q/cancel)")
    g.add_argument("--model", default=None, help="the executor's model folder, which every server of the queue must serve")
    g.add_argument("--generations", type=int, default=None,
                   help="number of finished generations the run folder should hold at the end (default 1): a repeated call runs nothing "
                        "more; raise it to run more; an unfinished generation is finished first")
    g.add_argument("--offline", action="store_true", default=None, help="write no request: reads that are not already in q/out count as missing")
    g.add_argument("--cards", type=int, default=None, help="proposer cards per generation (0 = no proposer)")
    g.add_argument("--config", default=None, help="yaml file deep-merged over the loop settings (config/loop.yaml); it may also hold the pipeline's loop keys "
                                                  "(queue_dir, generations, offline, proposer_cards, start_tip, start_sources, start_verdicts, stop_before) and model")
    g.add_argument("--base-model-config", default=None, help="yaml file whose base_model block replaces the one in basersi/config/pipeline.yaml; the "
                                                             "executor's model folder (or the stock folder its basersi_edit.json leads back to) is checked against it")
    g.add_argument("--start-tip", default=None, help="tip json {id, entries} for a new run directory (default: the stock model)")
    g.add_argument("--start-sources", default=None, help="json {entry id: [benchmark]} giving the source benchmark of start-tip entries")
    g.add_argument("--start-verdicts", default=None, help="json {entry id: verdict} with the layer of start-tip entries")
    g.add_argument("--accept-outputs-of-other-code", action="store_true", help="continue a run folder whose generations were made by other package code "
                                                                             "with the same inputs (recorded in state.json and printed); otherwise it is refused")
    g.add_argument("--stop-before", default=None, choices=["signals", "candidates", "prescreen", "judge", "update", "report"],
                   help="stop before this step (resumable)")
    i = sub.add_parser("init", help="write state.json of a new run directory from a start tip (an existing one is kept if it has the same start)",
                       epilog=config.EXIT_CODES, formatter_class=argparse.RawDescriptionHelpFormatter)
    i.add_argument("--run-dir", required=True, help="new run directory")
    i.add_argument("--tip", required=True, help="tip json {id, entries}")
    i.add_argument("--sources", default=None, help="json {entry id: [benchmark]}")
    i.add_argument("--verdicts", default=None, help="json {entry id: verdict}, or an acceptance-test file with entry_audit records (needs --members)")
    i.add_argument("--members", default=None, help="json {audit name: [entry id]} naming the entries of each entry_audit record")
    return p


def _json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise config.MissingInput(f"{path} does not exist") from None
    except ValueError as e:
        raise config.Refused(f"{path} is not a json file: {e}") from None


def _yaml(path):
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise config.MissingInput(f"{path} does not exist") from None
    except yaml.YAMLError as e:
        raise config.Refused(f"{path} is not a yaml file: {e}") from None


def main(argv=None):
    return config.run_command(_main, argv)


def _main(argv):
    a = parser().parse_args(argv)
    if a.command == "init":
        verdicts = None
        if a.verdicts:
            verdicts = verdicts_from_audit(a.verdicts, a.members) if a.members else _json(a.verdicts)
        sp = stage.prepare_run_dir(a.run_dir, config.load("loop"), _json(a.tip), _json(a.sources) if a.sources else None, verdicts)
        st = _json(sp)
        print(f"{sp}: tip {st['tip']['id']}, general {len(st['tip']['entries'])} entries, specialisation {len(st['tip']['special'])}")
        return 0
    lc = {}
    if a.config:
        lc = dict(_yaml(a.config) or {})
    given = dict(run_dir=a.run_dir, queue_dir=a.queue_dir, generations=a.generations, offline=a.offline,
                 proposer_cards=a.cards, start_tip=a.start_tip, start_sources=a.start_sources, start_verdicts=a.start_verdicts, stop_before=a.stop_before)
    lc.update({k: v for k, v in given.items() if v is not None})
    cfg_model = lc.pop("model", None)
    model = a.model or cfg_model
    if not model:
        raise Refuse("--model (the executor's model folder) is required")
    from basersi.pipeline import base_model_of
    out = stage.run({"work_dir": a.run_dir, "model": model, "base_model": base_model_of(a.base_model_config), "outputs": {"data": {"development_rows": a.rows}},
                     "loop": lc, FP.OTHER_CODE_KEY: bool(a.accept_outputs_of_other_code)})
    print(json.dumps(out, indent=1))
    return 0
