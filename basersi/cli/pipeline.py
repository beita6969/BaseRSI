import argparse
import json

from basersi import config as C
from basersi import pipeline as PL

HELP = "the whole process from a config: data, loop, acceptance of the entries, view edit, its acceptance, procedure edit, export (or one step)"


EPILOG = """inputs the config must name:
  model        folder of the stock base checkpoint; base_model names it (id, revision, config.json and index sha256,
               every weight shard's size and hashes) and view_edit / loop / procedure_edit / export and the judge's
               candidate reads refuse a folder that differs or that holds basersi_edit.json (an edited copy is never
               the base; a candidate checkpoint is traced back to its stock folder through its basersi_edit.json);
               base_model.weights_check: quick proves the shard names, sizes and the first MiB of each shard only (a
               byte changed later in a shard passes); full proves the content (sha256 of every byte, minutes per shard);
               without Hugging Face download metadata the revision is not checked and the result says so.
               A base_model block with an id in --config replaces the shipped one as a whole (another checkpoint);
               a block without id changes only the keys it gives (for example weights_check: full).
               It is checked before the first step that loads or serves a model
  data_dir     folder for the development rows; data.sources = folder of the official benchmark files
  loop.queue_dir, loop.generations (the TOTAL number of finished generations the run folder should hold, 8 in the shipped config, as in the paper;
               a repeated run with the same value runs nothing new, a larger value runs only the missing ones),
               loop.start_tip (a tip json to start from; null = a fresh start from the
               stock model with no entries; read only when the loop run folder holds no state.json yet; the
               folder is loop.run_dir, else work_dir/loop/<loop.run_name>, else work_dir/loop/main);
               the loop's own acceptance tests use the judge
               settings of this config (judge.yaml with judge.config_override applied), whereas `basersi loop
               generation` run by hand uses the shipped judge.yaml
  procedure_edit.strength.step: null (default: the line search over procedure_edit.yaml strength.steps, which takes the
               step with the lowest held-out divergence from the teacher) or a step that is used as given
  order:       the loop and the acceptance test of its entries run first; the View Edit is built only when the view
               entry (the image-token floor) is accepted, and the Procedure Edit only when the View Edit is

GPUs and ports (set CUDA_VISIBLE_DEVICES for each process yourself):
  loop      needs servers started by hand: `basersi serve server` (engine.yaml serve.ports.loop_server)
            and `basersi serve dispatcher --queue-dir <loop.queue_dir>`
  judge     starts its own server on judge.candidate_read_port with judge.candidate_read_cards
  procedure_edit  starts its own server on engine.yaml serve.ports.reads with procedure_edit.yaml serving.cards
  a port that serves another model is refused (the error names it); one serving the same checkpoint is reused

""" + C.EXIT_CODES + """
  3 here: pipeline_state.json "stopped" says why and which keys to set, including a Procedure Edit candidate
  NOT JUDGED because it generates more tokens than the reference frontier covers, and a Procedure Edit whose
  line search found no step with a finite divergence (no spec to export)

resume: work_dir/pipeline_state.json keeps a fingerprint of each step: its config section and the shipped yaml it
  reads (operational keys left out: ports, timeouts, concurrency, cards, card index, memory fraction, threads,
  the stage's own work_dir, wait and poll times; the device type cpu / cuda is kept), model and base_model
  (stages that use them), the outputs of the earlier steps it reads (view_edit: judge; loop: data; judge: data, view_edit, loop,
  export; procedure_edit: data, judge; export: view_edit, procedure_edit) and the code (version and a hash of the
  sources; for the steps that read rows, data / loop / judge / procedure_edit, also a hash of the shipped row
  manifest); for the steps that read through the engine (loop, judge,
  procedure_edit) also a digest of the engine.yaml read settings and the grading rules (benchmarks.yaml
  answer_match). While a step runs, the outputs of the later steps that read it are held: they come back when it
  ends with the same outputs, are cleared when its outputs changed, and stay held when it fails.
  Other code: by default a step whose code identity changed is re-run or refused like any other change. With
  resume.accept_outputs_of_other_code: true, a step whose fingerprint differs ONLY in the code identity keeps its
  outputs (and the judge its candidate reads); both code identities are written to pipeline_state.json
  (outputs_of_other_code, once per pair of identities) and to the step's record, with one line on stderr per
  reused output. Any other change (model, rows, the row manifest, config, read settings) is never relaxed by it.
  Every config key must be one the shipped yaml declares, with the shape it declares (mapping, list or scalar),
  the kind of value (number, text, true / false) and the length of a pair: an unknown key or any other value is
  refused with the valid keys (and a close match) or the valid form. Each stages item takes only stage and parts;
  a step may be listed once.
  view_edit / procedure_edit / export: an output is reused only under an identical fingerprint; otherwise it is
  rebuilt or refused.
  loop: the run folder is continued while the fingerprint is unchanged (loop.generations, loop.stop_before,
  loop.judge_exports_dir and loop.judge_export may change: the export is rebuilt from the cached reads and its
  settings key the judge step; raise loop.generations to add generations in the same work_dir); under a changed fingerprint
  (for example another queue_dir, start_tip or loop setting, other code) an existing run
  folder is refused with the mv that moves it away; set loop.run_name to keep both
  data / judge: re-run every time; the judge's candidate reads are re-read when the checkpoint, the rows, the
  read settings, the row manifest or the code changed (the old file is renamed *.superseded-<time>); missing or
  errored candidate reads are refused unless judge exports.allow_missing_reads is true
paths in --config are relative to the config file; paths in the shipped config to the working directory"""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="basersi pipeline", description=HELP, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="yaml file merged over basersi/config/pipeline.yaml (work_dir, data_dir, model, per-stage sections)")
    ap.add_argument("--only", default=None, help="run one step (stage name, step label such as 'judge[procedure_edit]', or step index)")
    ap.add_argument("--from", dest="start", default=None, help="first step to run (outputs of earlier steps are read from work_dir/pipeline_state.json)")
    ap.add_argument("--to", dest="stop", default=None, help="last step to run")
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    ap.add_argument("--no-gates", action="store_true", help="run view_edit and procedure_edit even when the view entry, the View Edit or the procedure entry was not accepted, and export without a spec check")
    a = ap.parse_args(argv)
    cfg = PL.load_config(a.config)
    if a.no_gates:
        cfg["acceptance_gates"] = False
    if a.list:
        for s in PL.steps(cfg):
            print(f"{s['index']}  {s['label']:32s} {PL.STAGE_MODULES[s['stage']]}.run")
        return 0
    res = PL.run(cfg, only=a.only, start=a.start, stop=a.stop)
    print(json.dumps(res, indent=1, default=str))
    return 3 if "stopped" in res else 0
