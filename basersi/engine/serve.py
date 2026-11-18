import json
import os
import subprocess
import time
import urllib.request

from basersi import config, fingerprint

_SERVE = config.load("engine")["serve"]


class ServerMismatch(config.Refused, RuntimeError):
    pass


def model_id(model_dir):
    root = os.path.realpath(model_dir)
    if not os.path.isdir(root):
        raise ServerMismatch(f"model folder {model_dir} does not exist")
    marker = fingerprint.read_marker(root)
    if marker is not None and (not marker.get("content_sha256") or marker.get("pending")):
        raise ServerMismatch(f"{root} is being patched in place ({fingerprint.MARKER} has no content_sha256 or a pending edit): it is not served or read. "
                             "Finish the patch (run the same 'basersi export patch' again) or serve another folder")
    return "m" + fingerprint.checkpoint_identity(root)[:16]


def served_name(model_dir, served_model_name=None):
    return served_model_name or _SERVE.get("served_model_name") or model_id(model_dir)


def image_limit():
    return int(_SERVE["max_images_per_prompt"])


def port_of(role):
    return int(_SERVE["ports"][role])


def vllm_command(model_dir, port, cards="one_card", served_model_name=None, max_num_seqs=None, gpu_memory_utilization=None, extra_args=()):
    layout = _SERVE[cards]
    return ["vllm", "serve", model_dir, "--served-model-name", served_name(model_dir, served_model_name),
            "--host", _SERVE["host"], "--port", str(port),
            "--tensor-parallel-size", str(layout["tensor_parallel_size"]), "--max-model-len", str(_SERVE["max_model_len"]),
            "--max-num-batched-tokens", str(_SERVE["max_num_batched_tokens"]), "--enable-chunked-prefill",
            "--kv-cache-dtype", _SERVE["kv_cache_dtype"],
            "--gpu-memory-utilization", str(gpu_memory_utilization if gpu_memory_utilization is not None else layout["gpu_memory_utilization"]),
            "--max-num-seqs", str(max_num_seqs if max_num_seqs is not None else layout["max_num_seqs"]), "--no-enable-prefix-caching",
            "--limit-mm-per-prompt", json.dumps({"image": int(_SERVE["max_images_per_prompt"]), "video": 0}),
            "--reasoning-parser", _SERVE["reasoning_parser"], "--seed", str(_SERVE["seed"]), "--disable-uvicorn-access-log"] + list(extra_args)


def served_models(port, host=None):
    try:
        with urllib.request.urlopen(f"http://{host or _SERVE['host']}:{port}/v1/models", timeout=5) as resp:
            data = json.loads(resp.read()).get("data", [])
    except Exception:
        return None
    return [dict(id=m.get("id"), root=m.get("root")) for m in data]


def _same_root(root, model_dir):
    return model_dir is None or (bool(root) and os.path.realpath(root) == os.path.realpath(model_dir))


def check_models(models, expected, model_dir=None, where=""):
    if any(m["id"] == expected and _same_root(m["root"], model_dir) for m in models):
        return True
    have = ", ".join(f"{m['id']} ({m['root'] or 'no model folder reported'})" for m in models) or "no model"
    raise ServerMismatch(f"{where or 'the server'} serves {have}, not {expected}" + (f" ({os.path.realpath(model_dir)})" if model_dir else "")
                         + ": stop that server or use another port")


def server_up(port, served_model_name=None, model_dir=None):
    models = served_models(port)
    if models is None:
        return False
    if served_model_name is None and model_dir is None:
        return True
    return check_models(models, served_name(model_dir, served_model_name), model_dir, where=f"port {port}")


def start_server(model_dir, port, cards="one_card", served_model_name=None, log_path=None, wait_seconds=None, **kw):
    wait_seconds = float(_SERVE["start_wait_seconds"] if wait_seconds is None else wait_seconds)
    name = served_name(model_dir, served_model_name)
    if server_up(port, name, model_dir):
        return None
    out = open(log_path, "a", encoding="utf-8") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(vllm_command(model_dir, port, cards, name, **kw), stdout=out, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONUNBUFFERED="1"))
    t0 = time.time()
    while time.time() - t0 < wait_seconds:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited with code {proc.returncode} (log: {log_path})")
        if server_up(port, name, model_dir):
            return proc
        time.sleep(10)
    proc.terminate()
    raise RuntimeError(f"vllm serve did not come up on port {port} within {wait_seconds:.0f} s")


def stop_server(proc, grace_seconds=None):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(float(_SERVE["stop_grace_seconds"] if grace_seconds is None else grace_seconds))
    except subprocess.TimeoutExpired:
        proc.kill()
