import asyncio
import glob
import heapq
import itertools
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from basersi import config, fingerprint
from basersi.engine import request as R
from basersi.engine import queue_file as QF
from basersi.engine import reads, router, serve
from basersi.engine.grader import grade_read, last_boxed
from basersi.engine.sampling import CHAT_MAX, CHAT_TEMP, READ_MAX, SAMPLING, cache_salt, crn_seed, read_body, thinking_kwargs

_ENGINE = config.load("engine")
_D = _ENGINE["dispatcher"]
ZOOM_MAX_TOKENS = int(_ENGINE["ops"]["zoom_max_tokens"])
DEFAULT_SERVER_IMAGE_LIMIT = int(_ENGINE["serve"]["max_images_per_prompt"])
CONTEXT_ERRORS = ("maximum context length", "longer than the maximum")
POOL = ThreadPoolExecutor(int(_D["threads"]))
S = None
Q = None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def proposed_answer(content):
    b = last_boxed(content or "")
    return b if b is not None else (content or "")[-300:]


class QueuePaths:
    def __init__(self, queue_dir, rows_path, tries):
        self.root, self.rows_path, self.tries = queue_dir, rows_path, tries
        self.qin, self.qout, self.qcan = (os.path.join(queue_dir, "q", d) for d in ("in", "out", "cancel"))
        self.drained = os.path.join(queue_dir, "q", "drained")
        self.engines = os.path.join(queue_dir, "engines.json")
        self.route_cache = os.path.join(queue_dir, "route_cache.jsonl")
        self.identity = os.path.join(queue_dir, "served_model.json")
        self.expected = os.path.join(queue_dir, "expected_model.json")
        for d in (self.qin, self.qout, self.qcan):
            os.makedirs(d, exist_ok=True)


class State:
    def __init__(self):
        self.done = set()
        self.batches = {}
        self.heap = []
        self.cv = asyncio.Condition()
        self.seq = itertools.count()
        self.rows, self.rows_mtime = {}, 0
        self.route = {}
        self.servers = {}
        self.queued_keys = set()
        self.engines_seen = False
        self.bad_batches = set()
        self.image_limit_count = 0

    def load_rows(self):
        f = Q.rows_path
        if os.path.getmtime(f) != self.rows_mtime:
            self.rows_mtime = os.path.getmtime(f)
            try:
                rows = reads.verified_rows(f)
                R.verify_images(rows.values())
                self.rows = rows
            except config.Refused as e:
                if not self.rows:
                    raise
                log(f"rows: {f} changed and is REFUSED ({e}); the verified rows loaded before are kept")
                return
            log(f"rows: {len(self.rows)} verified development rows from {f}")


def aborted(r):
    if r.get("error") or r.get("op_skipped") or r.get("mode") == "chat":
        return False
    return r.get("finish") == "abort" or (not r.get("content") and not r.get("reasoning") and not (r.get("tokens") or 0))


def index_out():
    bad = []
    note = lambda line, key, rescued: bad.append(key)
    for f in glob.glob(os.path.join(Q.qout, "*.jsonl")):
        for r in QF.read_records(f, note):
            if r.get("key") and "error" not in r and not aborted(r):
                S.done.add(r["key"])
    log(f"index: {len(S.done)} keys done in q/out" + (f"; {len(bad)} cut lines (their requests are made again)" if bad else ""))
    if os.path.exists(Q.route_cache):
        for r in QF.read_records(Q.route_cache, note):
            if "row" in r and "cond" in r:
                S.route[(r["row"], r["cond"], json.dumps(r.get("view")), r.get("rv", router.ROUTER_VERSION))] = r
    log(f"route cache: {len(S.route)}")


def out_write(batch, rec):
    QF.append_record(os.path.join(Q.qout, f"{batch}.jsonl"), rec)


def maybe_done(batch):
    b = S.batches[batch]
    if not b["closed"] and len(b["finished"]) + b["cancelled"] >= b["total"]:
        b["closed"] = True
        p = os.path.join(Q.qout, f"{batch}.done")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(dict(total=b["total"], finished=len(b["finished"]), cancelled=b["cancelled"], errors=b["errors"], t=time.time()), f)
        os.replace(p + ".tmp", p)
        log(f"batch {batch} done: {len(b['finished'])} results, {b['cancelled']} cancelled, {b['errors']} errors")


def _push(r, batch):
    heapq.heappush(S.heap, (int(r.get("priority", 5)), next(S.seq), batch, r))


async def scan_in():
    while not S.engines_seen:
        await asyncio.sleep(1)
    while True:
        try:
            S.load_rows()
            for f in sorted(glob.glob(os.path.join(Q.qin, "*.jsonl")), key=os.path.getmtime):
                batch = os.path.basename(f)[:-6]
                if batch in S.batches or batch in S.bad_batches:
                    continue
                try:
                    reqs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
                    if any(not isinstance(r, dict) or "key" not in r for r in reqs):
                        raise ValueError("a line that is not a request with a key")
                except (OSError, ValueError) as e:
                    S.bad_batches.add(batch)
                    log(f"batch {batch}: unreadable ({e!r:.200}); skipped")
                    continue
                closed = os.path.exists(os.path.join(Q.qout, f"{batch}.done"))
                if os.path.exists(os.path.join(Q.qcan, batch)):
                    S.batches[batch] = dict(total=len(reqs), finished=set(), cancelled=len(reqs), errors=0, closed=closed)
                    log(f"batch {batch}: cancelled ({len(reqs)} requests), not queued")
                    maybe_done(batch)
                    continue
                b = dict(total=len(reqs), finished=set(), cancelled=0, errors=0, closed=closed)
                S.batches[batch] = b
                new = 0
                for r in reqs:
                    k = r["key"]
                    if r.get("redo") and k in S.done and k not in S.queued_keys:
                        S.done.discard(k)
                        log(f"{k}: sent again on request of the loop (its answer line was cut)")
                    if k in S.done:
                        b["finished"].add(k)
                        continue
                    if k in S.queued_keys:
                        b.setdefault("mirror", []).append(k)
                        continue
                    S.queued_keys.add(k)
                    _push(r, batch)
                    new += 1
                    async with S.cv:
                        S.cv.notify_all()
                log(f"batch {batch}: {len(reqs)} requests, {len(b['finished'])} already done, {new} queued")
                maybe_done(batch)
            for batch, b in S.batches.items():
                if b.get("mirror"):
                    keep = []
                    for k in b["mirror"]:
                        if k in S.done:
                            b["finished"].add(k)
                        else:
                            keep.append(k)
                    b["mirror"] = keep
                    maybe_done(batch)
        except Exception:
            traceback.print_exc()
        await asyncio.sleep(float(_D["scan_seconds"]))


def build(r):
    body, meta = _build(r)
    body["cache_salt"] = cache_salt(r)
    return body, meta


def _build(r):
    mode = r.get("mode", "read")
    if mode == "chat":
        msgs, nt = R.chat_messages(r["messages"])
        body = dict(messages=msgs, max_tokens=int(r.get("max_tokens") or CHAT_MAX), chat_template_kwargs=thinking_kwargs(), **SAMPLING)
        body["temperature"] = CHAT_TEMP if r.get("temperature") is None else float(r["temperature"])
        if r.get("seed") is not None:
            body["seed"] = int(r["seed"])
        return body, dict(image_tokens=nt)
    row = S.rows[r["row"]]
    if mode == "route":
        body, nt = router.route_body(row, r["condition"], r.get("view"))
        return body, dict(image_tokens=nt)
    body, nt = read_body(row, r.get("settings"), r.get("prefix"), r.get("sample", 0), r.get("seed"), r.get("max_tokens"))
    return body, dict(image_tokens=nt, seed=body["seed"])


def graded_fields(row, r, content, reasoning, finish, tokens):
    ok, loop, gerr, pred = grade_read(row, content, reasoning, finish, tokens, r.get("sample", 0))
    return dict(ok=ok, loop=loop, grade_error=gerr, pred=pred, id=row["id"], kind=row["kind"], answer=row["answer"], options=row.get("options") or [],
                category=row.get("category"), precision=row.get("precision"), bench=row.get("bench") or row["id"].split("/")[0])


async def handle(srv, sess, r):
    import aiohttp
    key, mode = r["key"], r.get("mode", "read")
    t_sub = time.time()
    rec = dict(key=key, mode=mode)
    if mode in ("read", "route"):
        rec.update(row=r["row"])
    if mode == "read":
        rec["sample"] = r.get("sample", 0)
    if mode == "route":
        ch = router.condition_hash(r["condition"])
        rec["cond"] = ch
        hit = S.route.get((r["row"], ch, json.dumps(r.get("view")), router.ROUTER_VERSION))
        if hit:
            rec.update({k: hit.get(k) for k in ("yes", "p", "py", "pn", "p_src", "content", "tokens", "prompt_tokens")}, rv=router.ROUTER_VERSION,
                       view=r.get("view"), cached=True, t_submit=t_sub, t_done=time.time())
            return rec
    if mode == "read" and r.get("ops"):
        if r["row"] not in S.rows:
            rec.update(error=f"KeyError: row {r['row']} not in rows", server=srv, t_submit=t_sub, t_done=time.time())
            return rec
        return await run_ops(srv, sess, r, rec, t_sub)
    err, o, meta = None, None, {}
    for k in range(Q.tries):
        try:
            if mode != "chat" and r["row"] not in S.rows:
                raise KeyError(f"row {r['row']} not in the rows file")
            body, meta = await asyncio.get_running_loop().run_in_executor(POOL, build, r)
            body["model"] = S.servers[srv]["served"]
            async with sess.post(f"http://{srv}/v1/chat/completions", json=body) as resp:
                txt = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {txt[:300]}")
                o = json.loads(txt)
            break
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
            if isinstance(e, aiohttp.ClientConnectorError):
                rec.update(error=err, conn=True)
                return rec
            if isinstance(e, KeyError) or any(x in str(e) for x in CONTEXT_ERRORS):
                break
            if k + 1 < Q.tries:
                await asyncio.sleep(float(_D["retry_wait_seconds"]) * float(_D["retry_wait_growth"]) ** k)
    if o is None:
        rec.update(error=err, server=srv, t_submit=t_sub, t_done=time.time())
        return rec
    c = o["choices"][0]
    content, reasoning, finish, toks, ptoks = message_of(o)
    rec.update(server=srv, t_submit=t_sub, t_done=time.time())
    if mode == "route":
        content = content or reasoning
        res = dict(router.yes_probability(c, content), content=content, tokens=toks, prompt_tokens=ptoks)
        rec.update(res, rv=router.ROUTER_VERSION, view=r.get("view"), cached=False)
        cr = dict(row=r["row"], cond=rec["cond"], view=r.get("view"), rv=router.ROUTER_VERSION, condition=r["condition"], t=time.time(), **res)
        S.route[(r["row"], rec["cond"], json.dumps(r.get("view")), router.ROUTER_VERSION)] = cr
        QF.append_record(Q.route_cache, cr)
        return rec
    rec.update(content=content, reasoning=reasoning, finish=finish, tokens=toks)
    if mode == "chat":
        rec.update(prompt_tokens=ptoks, image_tokens=meta.get("image_tokens"))
        return rec
    row = S.rows[r["row"]]
    rec.update(prompt_tokens=ptoks, image_tokens=meta["image_tokens"], seed=meta["seed"], settings=r.get("settings"), prefix=r.get("prefix"),
               **graded_fields(row, r, content, reasoning, finish, toks))
    return rec


async def post(sess, srv, body):
    import aiohttp
    err = None
    body = dict(body, model=S.servers[srv]["served"])
    body.setdefault("cache_salt", os.urandom(16).hex())
    for k in range(Q.tries):
        try:
            async with sess.post(f"http://{srv}/v1/chat/completions", json=body) as resp:
                txt = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {txt[:300]}")
                return json.loads(txt), None
        except aiohttp.ClientConnectorError:
            raise
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
            if any(x in str(e) for x in CONTEXT_ERRORS):
                break
            if k + 1 < Q.tries:
                await asyncio.sleep(float(_D["retry_wait_seconds"]) * float(_D["retry_wait_growth"]) ** k)
    return None, err


def build_ops(r, ops, box):
    row = S.rows[r["row"]]
    st = r.get("settings") or {}
    paths = row.get("image_paths") or []
    content, nt = R.content_for(row["prompt"], paths, st.get("min_image_tokens"), st.get("max_image_tokens"), r.get("prefix"))
    extra, trace = [], {}
    for o in ops:
        if o["op"] == "zoom" and box:
            p, n = R.crop_part(paths[box[0] - 1], box, nt[box[0] - 1])
            extra.append(p)
            trace["zoom_tokens"] = n
        elif o["op"] == "tiles":
            k = int(o.get("n", 2))
            for i, pth in enumerate(paths):
                ts = R.tile_parts(pth, k, nt[i])
                extra += [p for p, _ in ts]
            trace["tiles"] = len(extra) - (1 if (box and any(x["op"] == "zoom" for x in ops)) else 0)
    content = R.insert_after_last_image(content, extra)
    body = dict(messages=[{"role": "user", "content": content}], max_tokens=int(r.get("max_tokens") or READ_MAX), chat_template_kwargs=thinking_kwargs(), **SAMPLING)
    if st.get("temperature") is not None:
        body["temperature"] = float(st["temperature"])
    body["seed"] = int(r["seed"]) if r.get("seed") is not None else crn_seed(r["row"], r.get("sample", 0))
    return body, nt, trace


def build_plain(r, tail_text):
    row = S.rows[r["row"]]
    st = r.get("settings") or {}
    content, nt = R.content_for(row["prompt"], row.get("image_paths") or [], st.get("min_image_tokens"), st.get("max_image_tokens"), r.get("prefix"))
    return list(content) + [{"type": "text", "text": tail_text}]


def message_of(o):
    c = o["choices"][0]
    m = c["message"]
    return ((m.get("content") or "").strip(), (m.get("reasoning_content") or m.get("reasoning") or "").strip(), c.get("finish_reason"),
            o["usage"]["completion_tokens"], o["usage"]["prompt_tokens"])


async def run_ops(srv, sess, r, rec, t_sub):
    import aiohttp
    ops = sorted(r["ops"], key=lambda o: ["zoom", "tiles", "check"].index(o["op"]))
    row = S.rows[r["row"]]
    loop_ = asyncio.get_running_loop()
    trace, total = [], 0
    why = await loop_.run_in_executor(POOL, R.ops_skip_reason, row, ops, r.get("settings"))
    if why:
        rec.update(op_skipped=why, ops=r["ops"], server=srv, t_submit=t_sub, t_done=time.time(), id=row["id"], bench=row.get("bench") or row["id"].split("/")[0])
        return rec
    n_img, _ = await loop_.run_in_executor(POOL, R.ops_need, row, ops, r.get("settings"))
    lim = int(S.servers[srv]["cfg"].get("image_limit", DEFAULT_SERVER_IMAGE_LIMIT))
    if n_img > lim:
        rec.update(error=f"server image limit {lim} < {n_img} images of the request: not readable on this server", image_limit=True, server=srv,
                   t_submit=t_sub, t_done=time.time())
        return rec
    try:
        box = None
        if any(o["op"] == "zoom" for o in ops):
            content = await loop_.run_in_executor(POOL, build_plain, r, R.ZOOM_Q)
            o, err = await post(sess, srv, dict(messages=[{"role": "user", "content": content}], max_tokens=ZOOM_MAX_TOKENS, temperature=0.0,
                                                chat_template_kwargs={"enable_thinking": False}))
            if o is None:
                raise RuntimeError(f"zoom pass: {err}")
            txt, _, _, tk, pt = message_of(o)
            total += tk
            box = R.parse_box(txt, len(row.get("image_paths") or []))
            trace.append(dict(op="zoom", tokens=tk, prompt_tokens=pt, reply=txt[:120], box=box, zoom_fail=box is None))
        body, nt, tr = await loop_.run_in_executor(POOL, build_ops, r, ops, box)
        o, err = await post(sess, srv, body)
        if o is None:
            raise RuntimeError(f"read pass: {err}")
        content, reasoning, finish, tk, pt = message_of(o)
        total += tk
        trace.append(dict(op="read", tokens=tk, prompt_tokens=pt, finish=finish, **tr))
        if any(x["op"] == "check" for x in ops):
            prop = proposed_answer(content)
            cc = await loop_.run_in_executor(POOL, build_plain, r, R.CHECK_Q.format(a=prop))
            cb = dict(messages=[{"role": "user", "content": cc}], max_tokens=int(r.get("max_tokens") or READ_MAX), chat_template_kwargs=thinking_kwargs(),
                      **SAMPLING, seed=body["seed"])
            o, err = await post(sess, srv, cb)
            if o is None:
                raise RuntimeError(f"check pass: {err}")
            ok0, _, _, pred0 = grade_read(row, content, reasoning, finish, tk, r.get("sample", 0))
            content, reasoning, finish, tk, pt2 = message_of(o)
            total += tk
            trace.append(dict(op="check", tokens=tk, prompt_tokens=pt2, finish=finish, proposed=str(prop)[:300], ok_before=ok0, pred_before=pred0))
    except aiohttp.ClientConnectorError as e:
        rec.update(error=f"{type(e).__name__}: {e}"[:300], conn=True)
        return rec
    except Exception as e:
        rec.update(error=f"ops: {e!r}"[:300], server=srv, t_submit=t_sub, t_done=time.time(), op_trace=trace)
        return rec
    rec.update(server=srv, t_submit=t_sub, t_done=time.time(), content=content, reasoning=reasoning, finish=finish, tokens=total,
               prompt_tokens=pt, image_tokens=nt, seed=body["seed"], settings=r.get("settings"), prefix=r.get("prefix"), ops=r["ops"], op_trace=trace,
               **graded_fields(row, r, content, reasoning, finish, total))
    return rec


async def pop(info):
    while True:
        async with S.cv:
            while True:
                if info["cfg"].get("state", "up") != "up":
                    return None, None
                if S.heap:
                    break
                try:
                    await asyncio.wait_for(S.cv.wait(), 10)
                except asyncio.TimeoutError:
                    pass
            pr, sq, batch, r = heapq.heappop(S.heap)
        if os.path.exists(os.path.join(Q.qcan, batch)):
            S.batches[batch]["cancelled"] += 1
            S.queued_keys.discard(r["key"])
            maybe_done(batch)
            continue
        return batch, r


async def _requeue(r, batch):
    async with S.cv:
        _push(r, batch)
        S.cv.notify_all()


async def worker(srv, slot):
    info = S.servers[srv]
    while True:
        if slot >= info["cap"] or info["cfg"].get("state", "up") != "up" or not info.get("model"):
            await asyncio.sleep(3)
            continue
        batch, r = await pop(info)
        if batch is None:
            continue
        info["inflight"] += 1
        try:
            rec = await handle(srv, info["sess"], r)
        except Exception as e:
            traceback.print_exc()
            rec = dict(key=r["key"], mode=r.get("mode", "read"), error=f"dispatcher: {e!r}"[:300], server=srv)
        info["inflight"] -= 1
        b = S.batches[batch]
        if rec.get("conn"):
            log(f"{srv} unreachable ({rec['error'][:80]}); {r['key']} requeued")
            info["model"] = None
            await _requeue(r, batch)
            await asyncio.sleep(float(_D["unreachable_wait_seconds"]))
            continue
        if aborted(rec):
            na = int(r.get("_aborts", 0)) + 1
            if na <= int(_D["abort_retries"]):
                log(f"{srv}: {r['key']} aborted (finish {rec.get('finish')!r}, {rec.get('tokens')} tokens); requeued ({na}/{_D['abort_retries']})")
                await _requeue(dict(r, _aborts=na), batch)
                await asyncio.sleep(float(_D["abort_wait_seconds"]))
                continue
            rec = dict(key=r["key"], mode=r.get("mode", "read"), row=r.get("row"), sample=r.get("sample"), server=srv,
                       error=f"aborted by the server {na} times (finish {rec.get('finish')!r})", finish=rec.get("finish"))
        out_write(batch, rec)
        S.queued_keys.discard(r["key"])
        if rec.get("image_limit"):
            S.image_limit_count += 1
            log(f"{r['key']}: {rec['error']} ({S.image_limit_count} such requests so far)")
        if "error" in rec:
            b["errors"] += 1
            info["errors"] += 1
            log(f"error {r['key']} on {srv}: {rec['error'][:200]}")
        else:
            S.done.add(r["key"])
            info["finished"] += 1
        b["finished"].add(r["key"])
        maybe_done(batch)


def _json_or_none(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def queue_identity():
    return _json_or_none(Q.identity)


def expected_record(model_dir):
    return dict(root=os.path.realpath(model_dir), model_id=serve.model_id(model_dir))


def check_queue_settings(queue_dir, record=True):
    path = os.path.join(queue_dir, "queue_settings.json")
    want = dict(read_settings_sha256=reads.read_settings_sha256(), read_settings=reads.read_settings())
    have = _json_or_none(path)
    if have is None:
        if not record:
            return want
        os.makedirs(queue_dir, exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(want, f, indent=1)
        os.replace(path + ".tmp", path)
        return want
    if have.get("read_settings_sha256") != want["read_settings_sha256"]:
        old, new = have.get("read_settings") or {}, want["read_settings"]
        diff = sorted(k for k in set(old) | set(new) if fingerprint.canonical(old.get(k)) != fingerprint.canonical(new.get(k)))
        raise serve.ServerMismatch(f"the requests in {queue_dir} were made with other engine.yaml read settings ({', '.join(diff)} differ, see {path}): "
                                   "its answers are never mixed with answers under the current settings; use a new queue directory")
    return have


def expect_model(queue_dir, model_dir, record=True):
    path = os.path.join(queue_dir, "expected_model.json")
    want = expected_record(model_dir)
    served = _json_or_none(os.path.join(queue_dir, "served_model.json"))
    if served is not None and os.path.realpath(served.get("root") or "") != want["root"]:
        raise serve.ServerMismatch(f"the reads in {queue_dir} come from {served.get('root')}, not from {want['root']}: use a new queue directory for another model")
    have = _json_or_none(path)
    if have is None:
        if not record:
            return want
        os.makedirs(queue_dir, exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(want, f)
        os.replace(path + ".tmp", path)
        return want
    if have.get("root") != want["root"]:
        raise serve.ServerMismatch(f"{path} names the folder {have.get('root')}, not {want['root']}: use a new queue directory for another model")
    if have.get("model_id") != want["model_id"]:
        raise serve.ServerMismatch(f"{want['root']} is now {want['model_id']}, but the reads of {queue_dir} were made from {have.get('model_id')} ({path}): "
                                   "the folder changed since; use a new queue directory")
    return have


DERIVED_NAME = re.compile(r"^m[0-9a-f]{16}$")


def start_identity(s, m):
    if s.get("model_id"):
        return s["model_id"]
    if not _ENGINE["serve"].get("served_model_name") and DERIVED_NAME.match(m["id"] or ""):
        return m["id"]
    return None


_FOLDER_IDS = {}


def folder_identity(root, every=60.0):
    t, mid = _FOLDER_IDS.get(root, (0.0, None))
    if time.time() - t >= every:
        try:
            mid = serve.model_id(root)
        except serve.ServerMismatch as e:
            mid = f"unusable ({e})"
        _FOLDER_IDS[root] = (time.time(), mid)
    return mid


def check_expected(srv, root, sid, exp):
    if os.path.realpath(root) != exp["root"]:
        raise serve.ServerMismatch(f"server {srv} serves {root}, but this queue expects {exp['root']} ({Q.expected})")
    if sid is None:
        raise serve.ServerMismatch(f"server {srv} serves {root} under an explicit name and no identity was recorded when it started: "
                                   "start it with 'basersi serve server --queue-dir', or serve the derived name")
    if sid != exp["model_id"]:
        raise serve.ServerMismatch(f"server {srv} was started on {sid}, but this queue expects {exp['model_id']} ({Q.expected}): restart the server")
    now = folder_identity(exp["root"])
    if now != exp["model_id"]:
        raise serve.ServerMismatch(f"{exp['root']} is now {now}, but this queue expects {exp['model_id']} ({Q.expected}): the folder changed since")


def identify(srv, s, models):
    want = s.get("model")
    if want:
        serve.check_models(models, want, where=f"server {srv}")
        m = next(m for m in models if m["id"] == want)
    elif len(models) == 1:
        m = models[0]
    else:
        raise serve.ServerMismatch(f"server {srv} serves {[m['id'] for m in models]}: name one with \"model\" in engines.json")
    if not m.get("root"):
        raise serve.ServerMismatch(f"server {srv} serves {m['id']} but reports no model folder: it cannot be checked")
    sid = start_identity(s, m)
    exp = _json_or_none(Q.expected)
    if exp is not None:
        check_expected(srv, m["root"], sid, exp)
    have = queue_identity()
    if have is None:
        with open(Q.identity + ".tmp", "w", encoding="utf-8") as f:
            json.dump(dict(id=m["id"], root=m["root"], model_id=sid), f)
        os.replace(Q.identity + ".tmp", Q.identity)
    elif have.get("id") != m["id"]:
        raise serve.ServerMismatch(f"server {srv} serves {m['id']} ({m['root']}), but the reads of this queue come from {have['id']} ({have.get('root')}): "
                                   "use a new queue directory for another model")
    return m["id"], m["root"], sid


def still_expected(srv, info):
    exp = _json_or_none(Q.expected)
    if exp is None:
        return None
    try:
        check_expected(srv, info.get("root") or "", info.get("sid"), exp)
    except serve.ServerMismatch as e:
        return str(e)
    return None


async def served_models(sess, srv):
    try:
        async with sess.get(f"http://{srv}/v1/models") as resp:
            if resp.status != 200:
                return None
            data = json.loads(await resp.text()).get("data", [])
    except Exception:
        return None
    return [dict(id=m.get("id"), root=m.get("root")) for m in data]


async def engines_loop():
    import aiohttp
    while True:
        try:
            with open(Q.engines, encoding="utf-8") as f:
                cfg = json.load(f)
            for s in cfg["servers"]:
                srv = f"{s.get('host', _ENGINE['serve']['host'])}:{s['port']}"
                cap = int(s.get("cap", _D["server_cap"]))
                if srv not in S.servers:
                    sess = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=float(_D["connect_timeout_seconds"]),
                                                                               sock_read=float(_D["read_timeout_seconds"])),
                                                 connector=aiohttp.TCPConnector(limit=0))
                    S.servers[srv] = dict(cfg=s, cap=cap, inflight=0, finished=0, errors=0, sess=sess, slots=0, model=None, served=None, root=None, sid=None, refused=None)
                    log(f"server {srv} card {s.get('card')} cap {cap} image_limit {s.get('image_limit', DEFAULT_SERVER_IMAGE_LIMIT)}")
                info = S.servers[srv]
                info["cfg"], info["cap"] = s, cap
                stale = still_expected(srv, info) if info["model"] else None
                if stale:
                    log(f"server {srv} REFUSED: {stale}; no new request is sent to it")
                    info["model"], info["refused"] = None, stale
                if not info["model"]:
                    models = await served_models(info["sess"], srv)
                    if models is not None:
                        try:
                            info["model"], info["root"], info["sid"] = identify(srv, s, models)
                            info["served"] = info["model"]
                            info["refused"] = None
                            log(f"server {srv} serves {info['model']}")
                        except serve.ServerMismatch as e:
                            if info["refused"] != str(e):
                                log(f"server {srv} REFUSED: {e}")
                            info["refused"] = str(e)
                st = s.get("state", "up")
                if st == "drain" and info["inflight"] == 0 and not info.get("drained"):
                    os.makedirs(Q.drained, exist_ok=True)
                    with open(os.path.join(Q.drained, f"{s.get('name', srv)}.json"), "w", encoding="utf-8") as f:
                        json.dump(dict(server=srv, card=s.get("card"), t=time.time(), finished=info["finished"]), f)
                    info["drained"] = True
                    log(f"server {srv} drained (card {s.get('card')}): no request in flight, nothing new sent; it may be stopped")
                if st == "up":
                    info["drained"] = False
                while info["slots"] < cap:
                    asyncio.ensure_future(worker(srv, info["slots"]))
                    info["slots"] += 1
        except Exception:
            traceback.print_exc()
        S.engines_seen = True
        await asyncio.sleep(float(_D["engines_seconds"]))


async def queue_log(every):
    while True:
        await asyncio.sleep(every)
        log(f"queued {len(S.heap)}; " + "; ".join(f"{s} inflight={i['inflight']} finished={i['finished']}" for s, i in S.servers.items()))


async def _main(every):
    global S
    S = State()
    S.load_rows()
    index_out()
    await asyncio.gather(engines_loop(), scan_in(), queue_log(every))


def serve_queue(queue_dir, rows_path, tries=None, log_every=60.0):
    global Q
    R.verify_images(reads.verified_rows(rows_path).values())
    Q = QueuePaths(queue_dir, rows_path, int(tries or _ENGINE["request_tries"]))
    check_queue_settings(queue_dir)
    asyncio.run(_main(log_every))
