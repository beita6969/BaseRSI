import hashlib
import json
import os

from basersi import config
from basersi import fingerprint as FP
from basersi.loop import report
from basersi.judge import judge as JV
from basersi.loop.generation import STEPS, LoopRun, Refuse, init_state, jdump
from basersi.loop.judge_export import JudgeExport

STAGE_KEYS = ("run_dir", "queue_dir", "generations", "run_name", "start_tip", "start_sources", "start_verdicts", "proposer_cards",
              "offline", "stop_before", "judge_exports_dir")


def development_rows(cfg):
    out = ((cfg.get("outputs") or {}).get("data") or {}).get("development_rows")
    if out:
        return out
    return os.path.join(cfg["data_dir"], config.load("benchmarks")["development"]["file"])


def _load_json(path):
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise config.MissingInput(f"{path} does not exist") from None
    except ValueError as e:
        raise config.Refused(f"{path} is not a json file: {e}") from None


def prepare_run_dir(run_dir, loop_config, start_tip=None, start_sources=None, start_verdicts=None):
    sp = os.path.join(run_dir, "state.json")
    if start_tip is None:
        return sp
    given = dict(start_tip=start_tip, start_sources=start_sources, start_verdicts=start_verdicts)
    loaded = {k: _load_json(v) if isinstance(v, str) else v for k, v in given.items()}
    tip, sources, verdicts = loaded["start_tip"], loaded["start_sources"], loaded["start_verdicts"]
    start = dict(tip_sha256=FP.digest(tip), sources_sha256=FP.digest(sources), verdicts_sha256=FP.digest(verdicts))
    if os.path.exists(sp):
        have = _load_json(sp).get("start")
        if have != start:
            what = "another start tip, sources or verdicts" if have else "no recorded start tip (it began from the stock model)"
            raise Refuse(f"{sp} exists and was started from {what}: it is never restarted from a new tip. Use a new run folder (this one is kept)")
        return sp
    st = init_state(tip, sources, verdicts, default_policy=loop_config["default_read_policy"])
    st["start"] = start
    os.makedirs(run_dir, exist_ok=True)
    jdump(sp, st)
    return sp


def run_dir_of(cfg):
    lc = cfg.get("loop") or {}
    return lc.get("run_dir") or os.path.join(cfg["work_dir"], "loop", lc.get("run_name") or "main")


def checked_executor(cfg):
    from basersi.export.stage import check_export_base
    from basersi.model_check import BaseModelError
    try:
        return check_export_base(cfg["model"], cfg.get("base_model") or config.load("pipeline")["base_model"])
    except (BaseModelError, FP.StaleOutput) as e:
        raise Refuse(f"the executor's model {cfg['model']}: {e}") from None


def run(cfg):
    lc = dict(cfg.get("loop") or {})
    stage = {k: lc.pop(k) for k in list(lc) if k in STAGE_KEYS}
    loop_config = config.load("loop", lc)
    if stage.get("proposer_cards") is not None:
        loop_config["proposer"]["cards_per_generation"] = int(stage["proposer_cards"])
    run_dir = run_dir_of(cfg)
    rows_path = development_rows(cfg)
    if not stage.get("queue_dir"):
        raise Refuse("the queue folder (--queue-dir, or loop.queue_dir in the config) is required")
    if not cfg.get("model"):
        raise Refuse("the executor's model folder (--model, or model in the config), which every server of the queue must serve, is required")
    checked_executor(cfg)
    prepare_run_dir(run_dir, loop_config, stage.get("start_tip"), stage.get("start_sources"), stage.get("start_verdicts"))
    queue_dir = stage["queue_dir"]
    lr = LoopRun(run_dir, rows_path, queue_dir, cfg["model"], loop_config=loop_config, offline=bool(stage.get("offline")),
                 judge_settings=JV.settings_of_pipeline(cfg),
                 accept_other_code=bool(cfg.get(FP.OTHER_CODE_KEY)))
    tip_before = lr.tip()["id"]
    target = 1 if stage.get("generations") is None else int(stage["generations"])
    finished = sum(len(x["done"]) == len(STEPS) for x in lr.state["gens"].values())
    ran = lr.run(target - finished, stop_before=stage.get("stop_before")) if target > finished else []
    tip = lr.tip()
    exports, summary_sha256 = None, None
    unfinished = [g for g, x in lr.state["gens"].items() if len(x["done"]) < len(STEPS)]
    finished = sum(len(x["done"]) == len(STEPS) for x in lr.state["gens"].values())
    if loop_config["judge_export"]["enabled"] and finished >= target and not unfinished:
        exports = stage.get("judge_exports_dir") or os.path.join(run_dir, "judge_exports")
        JudgeExport(lr, exports).build()
        with open(os.path.join(exports, "export_summary.json"), "rb") as f:
            summary_sha256 = hashlib.sha256(f.read()).hexdigest()
    accepted = [c["cid"] for g in ran for c in lr.state["gens"][str(g)].get("cands", []) if c.get("accept") and c.get("kind") != "audit"]
    return dict(run_dir=run_dir, state=os.path.join(run_dir, "state.json"), tip_id=tip["id"], tip=lr.write_tip(tip), tip_at_start=tip_before,
                general_entries=[e["cid"] for e in tip["entries"]],
                specialisation_entries=[dict(cid=e["cid"], source=e.get("source") or [], layer=(e.get("verdict") or {}).get("layer")) for e in tip["special"]],
                read_policy=dict(default=tip["policy"].get("default"), extras=[x["name"] for x in tip["policy"].get("extras", [])]),
                accepted=accepted, generations_run=ran, ledger=report.ledger_path(run_dir),
                reports=[report.report_path(run_dir, g) for g in ran if os.path.exists(report.report_path(run_dir, g))],
                queue_dir=queue_dir, origin=lr.origin, judge_exports=exports,
                judge_exports_summary_sha256=summary_sha256)
