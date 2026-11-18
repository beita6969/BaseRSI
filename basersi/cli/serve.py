import argparse
import json
import os
import signal

from basersi import config
from basersi.engine import dispatch, reads, request, serve

try:
    import fcntl
except ImportError:
    raise ImportError("basersi serve needs a POSIX system (Linux or macOS): it locks the queue's engines.json with fcntl") from None

HELP = "start a vLLM server for the executor, or the file-queue dispatcher that sends the loop's requests to the servers"
EPILOG = """\
The loop has one model, the executor. Its servers answer every request of the loop through one queue
folder: the reads, the route questions and the proposer's chats (the executor writes the new entries
for itself).

<queue-dir>/engines.json lists the servers: {"servers": [{name, host, port, cap, state, image_limit,
model, model_id, card}, ...]}. It is read again every few seconds.
  name         a label for the log
  host, port   where the server answers
  cap          requests in flight on this server
  state        up (send requests), drain (send nothing new; q/drained/<name>.json appears when idle),
               down (send nothing)
  image_limit  images per request the server was started with; a request with more images is
               recorded at once as not readable (an error line in q/out, counted in the log)
  model        optional served model name the server must report
  model_id     identity of the model folder when the server started (written by 'serve server' when it
               starts the server, or the derived served name, which is that identity); a server with a
               name given by hand and no model_id is refused
  card         optional label for the log
'serve server --queue-dir Q' adds or replaces its own entry with the image limit it was started
with; 'serve dispatcher --server host:port' writes entries only when the file does not exist yet.

Other files of the queue folder:
  expected_model.json  the model folder every server must serve (from 'serve dispatcher --model' or
                       from the loop's model); a server serving another folder is refused
  served_model.json    the served model the answers of this queue come from
  queue_settings.json  the engine.yaml read settings its requests were made under; a dispatcher or
                       loop with other read settings refuses this queue

A read whose only answer is an error is sent again when the dispatcher starts, and when the loop is
resumed (the loop waits for it before it writes the export): restart the dispatcher, then resume the
loop once.

""" + config.EXIT_CODES


def parser():
    p = argparse.ArgumentParser(prog="basersi serve", description=HELP, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("server", help="run vllm serve with the engine settings (blocks until stopped)", epilog=config.EXIT_CODES,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("--model", required=True, help="model directory: the stock checkpoint, or an edited checkpoint whose basersi_edit.json leads back to it; "
                                                   "the stock folder is checked against base_model before the server starts")
    s.add_argument("--base-model-config", default=None, help="yaml file whose base_model block replaces the one in basersi/config/pipeline.yaml")
    s.add_argument("--port", type=int, default=serve.port_of("loop_server"), help="default: engine.yaml serve.ports.loop_server")
    s.add_argument("--cards", default="one_card", choices=["one_card", "two_cards"], help="one_card = tensor parallel 1, two_cards = tensor parallel 2")
    s.add_argument("--served-model-name", default=None, help="default: an identifier derived from the model folder and its config")
    s.add_argument("--log", default=None, help="server log file")
    s.add_argument("--print-command", action="store_true", help="print the vllm command and exit")
    s.add_argument("--queue-dir", default=None, help="once up, add this server to <queue-dir>/engines.json with the image limit it was started with")
    s.add_argument("--host", default=None, help="host name written to engines.json (default: engine.yaml serve.host)")
    s.add_argument("--cap", type=int, default=None, help="requests in flight written to engines.json (default: engine.yaml dispatcher.server_cap)")
    d = sub.add_parser("dispatcher", help="serve the file queue: q/in requests -> servers listed in <queue-dir>/engines.json -> q/out results",
                       epilog=config.EXIT_CODES, formatter_class=argparse.RawDescriptionHelpFormatter)
    d.add_argument("--queue-dir", required=True, help="queue folder shared with the loop (q/in, q/out, engines.json, ...)")
    d.add_argument("--rows", required=True, help="development rows file (with prompts and image paths); checked against the shipped row manifest")
    d.add_argument("--server", action="append", default=[], help="host:port of a server to add to engines.json when the file does not exist (repeatable); "
                   "its image_limit is engine.yaml serve.max_images_per_prompt, the limit 'serve server' starts vLLM with")
    d.add_argument("--model", default=None, help="model folder every server must serve (recorded in <queue-dir>/expected_model.json)")
    d.add_argument("--cap", type=int, default=int(config.load("engine")["dispatcher"]["server_cap"]), help="requests in flight per server for --server entries")
    d.add_argument("--log-every", type=float, default=60.0)
    return p


def engine():
    return config.load("engine")


def register(queue_dir, entry):
    os.makedirs(queue_dir, exist_ok=True)
    engines = os.path.join(queue_dir, "engines.json")
    with open(engines + ".lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cfg = dict(servers=[])
        if os.path.exists(engines):
            with open(engines, encoding="utf-8") as f:
                cfg = json.load(f)
        cfg["servers"] = [s for s in cfg.get("servers", []) if not (s.get("host") == entry["host"] and int(s.get("port", -1)) == entry["port"])] + [entry]
        with open(engines + ".tmp", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
        os.replace(engines + ".tmp", engines)
    print(f"{engines}: {entry['name']} registered (image_limit {entry['image_limit']})")


def checked_model(model_dir, base_model_config=None):
    from basersi import fingerprint
    from basersi.export.stage import check_export_base
    from basersi.model_check import BaseModelError
    from basersi.pipeline import base_model_of
    try:
        return check_export_base(model_dir, base_model_of(base_model_config))
    except (BaseModelError, fingerprint.StaleOutput) as e:
        raise serve.ServerMismatch(f"model {model_dir}: {e}") from None


def main(argv=None):
    return config.run_command(_main, argv)


def _main(argv):
    a = parser().parse_args(argv)
    if a.command == "server":
        if a.print_command:
            print(" ".join(serve.vllm_command(a.model, a.port, a.cards, a.served_model_name)))
            return 0
        checked_model(a.model, a.base_model_config)
        started_on = serve.model_id(a.model)
        name = serve.served_name(a.model, a.served_model_name)
        proc = serve.start_server(a.model, a.port, a.cards, a.served_model_name, log_path=a.log)
        if proc is None and name != started_on:
            raise serve.ServerMismatch(f"port {a.port} already serves {name}, a name given by hand (--served-model-name or engine.yaml serve.served_model_name), "
                                       "so what that server loaded cannot be told from its name; it is not registered. Stop it and start it with this command")
        if a.queue_dir:
            register(a.queue_dir, dict(name=f"{a.host or engine()['serve']['host']}:{a.port}", host=a.host or engine()["serve"]["host"], port=a.port,
                                       cap=int(a.cap if a.cap is not None else engine()["dispatcher"]["server_cap"]), state="up",
                                       image_limit=serve.image_limit(), model=name, model_id=started_on, card=os.environ.get("CUDA_VISIBLE_DEVICES")))
        if proc is None:
            print(f"port {a.port} already serves {name}")
            return 0
        print(f"server up on port {a.port} (pid {proc.pid}), serving {name}")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGTERM)
            serve.stop_server(proc)
        return proc.returncode or 0
    request.verify_images(reads.verified_rows(a.rows).values())
    dispatch.check_queue_settings(a.queue_dir)
    engines = os.path.join(a.queue_dir, "engines.json")
    if a.server and not os.path.exists(engines):
        servers = []
        for k, hp in enumerate(a.server):
            host, port = hp.rsplit(":", 1)
            servers.append(dict(name=f"server{k}", host=host, port=int(port), cap=a.cap, state="up", image_limit=serve.image_limit()))
        with open(engines + ".tmp", "w", encoding="utf-8") as f:
            json.dump(dict(servers=servers), f, indent=1)
        os.replace(engines + ".tmp", engines)
    if not os.path.exists(engines):
        raise config.MissingInput(f"{engines} is missing: pass --server host:port or write it")
    if a.model:
        dispatch.expect_model(a.queue_dir, a.model)
    dispatch.serve_queue(a.queue_dir, a.rows, log_every=a.log_every)
    return 0
