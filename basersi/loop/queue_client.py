import glob
import json
import os
import time

from basersi import config
from basersi.engine import queue_file as QF


class QueueClient:
    def __init__(self, root, log=print, kept_tail_chars=None, offline=False):
        self.root, self.log, self.offline = root, log, bool(offline)
        self.kept_tail = int(config.load("loop")["kept_tail_chars"] if kept_tail_chars is None else kept_tail_chars)
        for d in ("q/in", "q/out", "q/cancel"):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        self.res, self.err, self.off, self.pending = {}, {}, {}, set()
        self.request_file, self.redone = {}, set()
        for f in sorted(glob.glob(os.path.join(root, "q/in/*.jsonl"))):
            if os.path.exists(os.path.join(root, "q/cancel", os.path.basename(f)[:-6])):
                continue
            for q in QF.read_records(f):
                if q.get("key"):
                    self.pending.add(q["key"])
                    self.request_file.setdefault(q["key"], f)
        self.poll()
        self.errored_before = set() if self.offline else set(self.err)

    def _bad_line(self, path, line, key, rescued):
        self.log(f"  {path}: a line cut by an interruption ({len(line)} bytes, key {key!r}); {len(rescued)} answer recovered after it")
        if key and key not in self.res and key not in self.redone and key not in [r.get("key") for r in rescued]:
            self.redone.add(key)
            if self.offline:
                self.log(f"  {key}: not sent again (offline: no request is written)")
                return
            q = self._request_from_file(key)
            if q is not None:
                self._write_batch("redo", [dict(q, redo=True)])
                self.log(f"  {key}: sent again at once")

    def _request_from_file(self, key):
        f = self.request_file.get(key)
        if f is None:
            return None
        return next((q for q in QF.read_records(f) if q.get("key") == key), None)

    def poll(self):
        n, bad = 0, []
        for f in sorted(glob.glob(os.path.join(self.root, "q/out/*.jsonl"))):
            o = self.off.get(f, 0)
            try:
                if os.path.getsize(f) <= o:
                    continue
                with open(f, "rb") as fh:
                    fh.seek(o)
                    buf = fh.read()
            except OSError:
                continue
            recs, end = QF.records(buf, lambda line, key, rescued, f=f: bad.append((f, line, key, rescued)))
            for r in recs:
                k = r.get("key")
                if not k:
                    continue
                if r.get("error"):
                    if k not in self.res:
                        self.err[k] = r
                    continue
                if r.get("finish") == "abort" or k in self.res:
                    continue
                for field in ("content", "reasoning"):
                    if field == "content" and r.get("mode") == "chat":
                        continue
                    if isinstance(r.get(field), str) and len(r[field]) > self.kept_tail:
                        r[field] = r[field][-self.kept_tail:]
                self.res[k] = r
                self.err.pop(k, None)
                n += 1
            self.off[f] = o + end
        for f, line, key, rescued in bad:
            self._bad_line(f, line, key, rescued)
        return n

    def have(self, k):
        return k in self.res

    def get(self, k):
        return self.res.get(k)

    def submit(self, name, reqs):
        self.poll()
        new, seen, again = [], set(), []
        for q in reqs:
            k = q["key"]
            if k in self.res or k in seen:
                continue
            if k in self.errored_before:
                self.errored_before.discard(k)
                self.err.pop(k, None)
                again.append(k)
            elif k in self.pending:
                continue
            seen.add(k)
            new.append(q)
        if not new:
            return 0
        if again:
            self.log(f"  {name}: {len(again)} reads whose only answer was an error are requested again and awaited")
        self._write_batch(name, new)
        self.pending.update(seen)
        return len(new)

    def _write_batch(self, name, new):
        stamp = int(time.time() * 1000)
        p = os.path.join(self.root, "q/in", f"{name}_{stamp}.jsonl")
        while os.path.exists(p) or os.path.exists(p + ".tmp"):
            stamp += 1
            p = os.path.join(self.root, "q/in", f"{name}_{stamp}.jsonl")
        with open(p + ".tmp", "w", encoding="utf-8") as fh:
            for q in new:
                fh.write(json.dumps(q, ensure_ascii=False) + "\n")
        os.replace(p + ".tmp", p)
        for q in new:
            self.request_file[q["key"]] = p

    def wait(self, keys, deadline=None, every=5.0, label="", tick=300):
        keys = list(dict.fromkeys(keys))
        t0, last = time.time(), time.time()
        while True:
            self.poll()
            miss = [k for k in keys if k not in self.res and k not in self.err]
            if not miss:
                return [k for k in keys if k in self.res], []
            if deadline is not None and time.time() >= deadline:
                return [k for k in keys if k in self.res], miss
            if time.time() - last >= tick:
                last = time.time()
                self.log(f"  waiting {label}: {len(keys) - len(miss)}/{len(keys)} done after {time.time() - t0:.0f}s")
            time.sleep(every)
