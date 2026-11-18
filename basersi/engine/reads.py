import asyncio
import hashlib
import json
import os
import time

from basersi import config
from basersi.data import manifest
from basersi.data.rows import ROLE_FIELD, ROLES, RowsError, check_field_names
from basersi.engine import request as R
from basersi.engine import serve
from basersi.engine.grader import grade_read
from basersi.engine.sampling import read_body

_READS = config.load("engine")["reads"]
READ_TRIES = int(_READS["tries"])


def read_settings(engine=None, direct=False):
    return config.read_settings(engine, direct=direct)


def read_settings_sha256(engine=None, direct=False):
    return config.read_settings_sha256(engine, direct=direct)


class RowRefused(config.Refused, ValueError):
    pass


def verified_rows(rows_path):
    try:
        with open(rows_path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        raise config.MissingInput(f"{rows_path}: the rows file does not exist") from None
    try:
        rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    except ValueError as e:
        raise RowRefused(f"{rows_path}: a line is not json ({e})") from None
    if any(not isinstance(r, dict) or "id" not in r for r in rows):
        raise RowRefused(f"{rows_path}: a line is not a row with an id")
    seen = set()
    twice = [r["id"] for r in rows if r["id"] in seen or seen.add(r["id"])]
    if twice:
        raise RowRefused(f"{rows_path}: {len(twice)} row ids appear more than once (e.g. {twice[0]!r})")
    try:
        check_field_names(rows, rows_path)
    except RowsError as e:
        raise RowRefused(str(e)) from None
    bench = config.load("benchmarks")
    if hashlib.sha256(raw).hexdigest() != bench["development"]["sha256"] and not manifest.same_rows(rows, bench):
        raise RowRefused(f"{rows_path} is not the development rows file: its rows (ids, roles, content) do not match the shipped row manifest. "
                         "Build it with 'basersi data'")
    foreign = [r["id"] for r in rows if r.get(ROLE_FIELD) not in ROLES]
    if foreign:
        raise RowRefused(f"{rows_path}: {len(foreign)} rows are not development rows (e.g. {foreign[0]!r})")
    images = R.manifest_images()
    try:
        R.register_row_images([dict(id=r["id"], image_paths=r.get("image_paths") or [], img_sha256=images[r["id"]]) for r in rows])
    except R.ImageRefused as e:
        raise RowRefused(str(e)) from None
    return {r["id"]: r for r in rows}


def development_rows(rows_path, row_ids):
    rows = verified_rows(rows_path)
    missing = [i for i in row_ids if i not in rows]
    if missing:
        raise RowRefused(f"{len(missing)} row ids are not in the development rows file {rows_path} (e.g. {missing[0]!r})")
    return rows


def request_parts(request, row_id):
    request = request or {}
    prefix_rows = request.get("prefix_rows")
    prefix = request.get("prefix") if (prefix_rows is None or row_id in set(prefix_rows)) else None
    return dict(request.get("settings") or {}), prefix


def read_request_body(row, sample, request, served_model_name):
    settings, prefix = request_parts(request, row["id"])
    body, image_tokens = read_body(row, settings, prefix, sample)
    body["model"] = served_model_name
    body["cache_salt"] = os.urandom(16).hex()
    return body, image_tokens


def _canon(x):
    return json.dumps(x, sort_keys=True)


def _close_last_line(out_path):
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return None
    with open(out_path, "rb") as f:
        data = f.read()
    if data.endswith(b"\n"):
        return None
    cut = data.rfind(b"\n") + 1
    tail = data[cut:]
    try:
        json.loads(tail)
    except ValueError:
        aside = f"{out_path}.incomplete_line.{time.strftime('%Y%m%d-%H%M%S')}.{os.getpid()}"
        with open(aside, "wb") as f:
            f.write(tail)
        with open(out_path, "r+b") as f:
            f.truncate(cut)
        print(f"{out_path}: the last line was cut by an interruption; its {len(tail)} bytes moved to {aside}, that read is made again", flush=True)
        return aside
    with open(out_path, "ab") as f:
        f.write(b"\n")
    return None


def _done(out_path, request=None):
    have = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if not isinstance(x, dict) or "row" not in x or "sample" not in x:
                continue
            if "req" in x and _canon(x["req"]) != _canon(request):
                raise config.Refused(f"{out_path} holds reads of another request ({_canon(x['req'])[:120]}): use a new output file")
            if not x.get("error") and x.get("finish") != "abort":
                have.add((x["row"], x["sample"]))
    return have


def _check_model_record(out_path, served, model_dir):
    meta = out_path + ".model.json"
    want = dict(served_model_name=served, model=os.path.realpath(model_dir), read_settings_sha256=read_settings_sha256(direct=True), read_settings=read_settings(direct=True))
    if os.path.exists(meta) and os.path.exists(out_path):
        with open(meta, encoding="utf-8") as f:
            have = json.load(f)
        what = [k for k in ("served_model_name", "model", "read_settings_sha256") if have.get(k) != want[k]]
        if what:
            raise serve.ServerMismatch(f"{out_path} holds reads made with another {' and '.join(what)} (see {meta}): use a new output file")
        return
    with open(meta + ".tmp", "w", encoding="utf-8") as f:
        json.dump(want, f)
    os.replace(meta + ".tmp", meta)


async def _read_one(sess, url, row, sample, request, served_model_name, fo):
    body, image_tokens = read_request_body(row, sample, request, served_model_name)
    rec = dict(row=row["id"], sample=sample, bench=row.get("bench"), seed=body["seed"], image_tokens=image_tokens, req=request)
    err, o = None, None
    for k in range(READ_TRIES):
        try:
            async with sess.post(url, json=body) as resp:
                txt = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {txt[:300]}")
                o = json.loads(txt)
                break
        except Exception as e:
            err = repr(e)[:300]
            if "HTTP 400" in err:
                break
            await asyncio.sleep(float(_READS["retry_wait_seconds"]) * (k + 1))
    if o is None:
        rec.update(error=err)
    else:
        c = o["choices"][0]
        m = c["message"]
        if c.get("finish_reason") == "abort":
            rec.update(error="finish abort")
        else:
            content = (m.get("content") or "").strip()
            reasoning = (m.get("reasoning_content") or m.get("reasoning") or "").strip()
            tokens, prompt_tokens, finish = o["usage"]["completion_tokens"], o["usage"]["prompt_tokens"], c.get("finish_reason")
            ok, loop, grade_error, pred = grade_read(row, content, reasoning, finish, tokens, sample)
            rec.update(ok=ok, pred=pred, loop=loop, grade_error=grade_error, tokens=tokens, prompt_tokens=prompt_tokens, finish=finish,
                       content=content, reasoning=reasoning, answer=row["answer"])
    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fo.flush()


async def _read_all(todo, rows, request, url, served_model_name, out_path, concurrency):
    import aiohttp
    queue = asyncio.Queue()
    for t in todo:
        queue.put_nowait(t)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=float(_READS["timeout_seconds"])), connector=aiohttp.TCPConnector(limit=0)) as sess:
        with open(out_path, "a", encoding="utf-8") as fo:
            async def worker():
                while not queue.empty():
                    row_id, sample = queue.get_nowait()
                    await _read_one(sess, url, rows[row_id], sample, request, served_model_name, fo)
            await asyncio.gather(*[worker() for _ in range(concurrency)])


def read_rows(model_dir, row_ids, samples, request, out_path, rows_path=None, port=None, cards="one_card", served_model_name=None,
              concurrency=None, connect_only=False, server_log=None):
    if not rows_path:
        raise config.Refused("rows_path (the development rows file) is required")
    rows = development_rows(rows_path, list(row_ids))
    try:
        R.verify_images(rows[i] for i in row_ids)
    except R.ImageRefused as e:
        raise RowRefused(str(e)) from None
    port = serve.port_of("reads") if port is None else int(port)
    concurrency = int(_READS["concurrency"] if concurrency is None else concurrency)
    served = serve.served_name(model_dir, served_model_name)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    _check_model_record(out_path, served, model_dir)
    _close_last_line(out_path)
    have = _done(out_path, request)
    todo = [(r, int(s)) for s in samples for r in row_ids if (r, int(s)) not in have]
    proc = None
    if todo and connect_only:
        if not serve.server_up(port, served, model_dir):
            raise RuntimeError(f"no server answers on port {port} (connect_only)")
    elif todo:
        proc = serve.start_server(model_dir, port, cards, served, log_path=server_log)
    try:
        if todo:
            url = f"http://{config.load('engine')['serve']['host']}:{port}/v1/chat/completions"
            t0 = time.time()
            asyncio.run(_read_all(todo, rows, request, url, served, out_path, concurrency))
            left = [t for t in todo if t not in _done(out_path, request)]
            print(f"{len(todo)} reads in {time.time() - t0:.0f} s -> {out_path}" + (f"; {len(left)} ended with an error and are retried by the next call" if left else ""),
                  flush=True)
    finally:
        serve.stop_server(proc)
    return out_path
