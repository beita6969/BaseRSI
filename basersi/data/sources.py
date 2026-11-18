import base64
import collections
import glob
import hashlib
import io
import json
import os
import re
import zipfile

from basersi import config
from basersi.data import manifest as M
from basersi.data import rows as RW

class SourceError(config.Refused, ValueError):
    pass


def repo_dir(sources_dir, repo):
    return os.path.join(sources_dir, repo.replace("/", "__"))


def check_files(sources_dir, scfg, benches, check_sha=True):
    out = {}
    for b in benches:
        d = scfg["datasets"][b]
        for f in d["files"]:
            p = os.path.join(repo_dir(sources_dir, d["repo"]), f["path"])
            if not os.path.exists(p):
                raise SourceError(f"{b}: missing {p}: download revision {d['revision']} of {d['url']} so that the file sits at {scfg['layout']}")
            if check_sha and f.get("bytes") is not None and os.path.getsize(p) != int(f["bytes"]):
                raise SourceError(f"{b}: {p} has {os.path.getsize(p)} bytes, expected {f['bytes']} (revision {d['revision']} of {d['repo']})")
            if check_sha:
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for blk in iter(lambda: fh.read(1 << 22), b""):
                        h.update(blk)
                if h.hexdigest() != f["sha256"]:
                    raise SourceError(f"{b}: {p} has sha256 {h.hexdigest()}, expected {f['sha256']} (revision {d['revision']})")
            out[f"{b}:{f['path']}"] = p
    return out


def dhash256(im):
    from PIL import Image
    g = im.convert("L").resize((17, 16), Image.LANCZOS)
    px = list(g.get_flattened_data()) if hasattr(g, "get_flattened_data") else list(g.getdata())
    bits = [px[r * 17 + c] > px[r * 17 + c + 1] for r in range(16) for c in range(16)]
    n = sum(bits)
    if n < 16 or n > 240:
        return None
    return "%064x" % int("".join("1" if b else "0" for b in bits), 2)


class Images:
    def __init__(self, images_dir):
        self.dir = images_dir

    def save(self, bench, key, data):
        from PIL import Image
        limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            return self._save(bench, key, data)
        finally:
            Image.MAX_IMAGE_PIXELS = limit

    def _save(self, bench, key, data):
        from PIL import Image
        if isinstance(data, dict):
            data = data.get("bytes")
        if isinstance(data, str):
            data = base64.b64decode(data)
        im = Image.open(io.BytesIO(data))
        fmt = (im.format or "").upper()
        if fmt not in ("JPEG", "PNG"):
            im.seek(0)
            im2 = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
            buf = io.BytesIO()
            im2.save(buf, "PNG")
            data = buf.getvalue()
            fmt = "PNG"
        ext = "jpg" if fmt == "JPEG" else "png"
        os.makedirs(os.path.join(self.dir, bench), exist_ok=True)
        key = re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))
        path = os.path.join(self.dir, bench, f"{key}.{ext}")
        sha = hashlib.sha256(data).hexdigest()
        n = 1
        while os.path.exists(path) and hashlib.sha256(open(path, "rb").read()).hexdigest() != sha:
            n += 1
            path = os.path.join(self.dir, bench, f"{key}__dup{n}.{ext}")
        if not os.path.exists(path):
            with open(path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(path + ".tmp", path)
        im.seek(0)
        return path, sha, dhash256(im)


class Builder:
    def __init__(self, sources_dir, images_dir, scfg):
        self.src = sources_dir
        self.s = scfg
        self.img = Images(images_dir)

    def path(self, bench, pattern):
        d = self.s["datasets"][bench]
        hits = sorted(glob.glob(os.path.join(repo_dir(self.src, d["repo"]), pattern), recursive=True))
        if not hits:
            raise SourceError(f"{bench}: no file {pattern} under {repo_dir(self.src, d['repo'])}")
        return hits

    def row(self, bench, sid, kind, text, imgs, answer, options=None, category=None, group=None, imgkeys=None, suffix=None, **extra):
        paths, shas, dh = [], [], []
        for k, b in enumerate(imgs):
            p, s, d = self.img.save(bench, imgkeys[k] if imgkeys else f"{sid}_{k}", b)
            paths.append(p)
            shas.append(s)
            dh.append(d)
        prompt = text.rstrip() + "\n\n" + (self.s["suffix"][kind] if suffix is None else suffix)
        r = {"id": f"{bench}/{sid}", "bench": bench, "kind": kind, "prompt": prompt, "image_paths": paths, "answer": str(answer), "options": options or [],
             "category": category, "group": group, "img_sha256": shas, "img_dhash": dh}
        r.update(extra)
        return r

    def items_babyvision(self):
        import pandas as pd
        df = pd.read_parquet(self.path("babyvision", "data/train-00000-of-00001.parquet")[0])
        for r in df.itertuples(index=False):
            yield str(r.taskId), r

    def build_babyvision(self, r):
        let, bv = self.s["letters"], self.s["babyvision_answer_line"]
        cat = f"{r.type}/{r.subtype}"
        if r.ansType == "choice":
            opts = list(r.options)
            q = r.question + "\nChoices:\n" + "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(opts)).strip() + bv
            row = self.row("babyvision", r.taskId, "mc", q, [r.image], let[int(r.choiceAns)], opts, cat, suffix="")
        else:
            g = canon_gold(r.blankAns)
            k = blank_kind(g)
            if k is None:
                raise SourceError(f"babyvision/{r.taskId}: free-text blank answer, not a development row")
            row = self.row("babyvision", r.taskId, k, r.question + bv, [r.image], g, None, cat, gold_raw=r.blankAns, suffix="")
        row["prompt"] = row["prompt"][:-2] if row["prompt"].endswith("\n\n") else row["prompt"]
        return row

    def items_blink(self):
        import pandas as pd
        for c in self.s["blink_configs"]:
            df = pd.read_parquet(self.path("blink", f"{c}/val-00000-of-00001.parquet")[0])
            for r in df.itertuples(index=False):
                yield str(r.idx), (c, r)

    def build_blink(self, item):
        c, r = item
        imgs = [x for x in (r.image_1, r.image_2, r.image_3, r.image_4) if x is not None and x.get("bytes")]
        text = re.sub(r"\s*Answer with the option'?s letter from the given choices directly\.?\s*$", "", r.prompt.strip())
        return self.row("blink", r.idx, "mc", text, imgs, r.answer.strip("() "), list(r.choices), c)

    def items_charxiv(self):
        import pandas as pd
        df = pd.read_parquet(self.path("charxiv", "val.parquet")[0])
        rs = list(df.itertuples(index=False))
        oid = [str(r.original_id) if r.original_id else str(n) for n, r in enumerate(rs)]
        c = collections.Counter(oid)
        for n, r in enumerate(rs):
            fig = os.path.splitext(os.path.basename(str(r.figure_path)))[0]
            yield (oid[n] if c[oid[n]] == 1 else f"{oid[n]}_f{fig}"), (r, fig)

    def build_charxiv(self, item, sid):
        r, fig = item
        t = int(r.reasoning_a_type)
        a = r.reasoning_a.strip()
        inst = self.s["charxiv_instructions"][t]
        q = inst.format(r.reasoning_q.strip(), number_instruction(a)) if t == 4 else inst.format(r.reasoning_q.strip())
        kind = "num" if t in (3, 4) else "short"
        return self.row("charxiv", sid, kind, q, [r.image], a, None, self.s["charxiv_categories"][t], group=r.original_figure_path, imgkeys=[f"fig_{fig}"],
                        suffix=self.s["suffix"]["num"] if kind == "num" else self.s["charxiv_short_suffix"], cx_type=t, figure_path=str(r.figure_path))

    def items_visulogic(self):
        z = zipfile.ZipFile(self.path("visulogic", "**/images.zip")[0])
        for l in open(self.path("visulogic", "**/data.jsonl")[0], encoding="utf-8"):
            x = json.loads(l)
            yield str(x["id"]), (x, z)

    def build_visulogic(self, item):
        x, z = item
        row = self.row("visulogic", x["id"], "mc", x["question"] + "\n" + self.s["visulogic_cot"], [z.read(x["image_path"])], x["label"], [], x["tag"], suffix="")
        row["prompt"] = row["prompt"][:-2] if row["prompt"].endswith("\n\n") else row["prompt"]
        return row

    def items_vlmsbiased(self):
        import pandas as pd
        files = self.path("vlmsbiased", "**/main-*.parquet")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        for r in df.itertuples(index=False):
            yield str(r.ID), r

    def build_vlmsbiased(self, r):
        text = re.split(r"\s*(?:Answer in curly|Answer with|Answer in)", r.prompt)[0].strip()
        gt = str(r.ground_truth).strip()
        if re.fullmatch(r"\d+", gt):
            kind = "num"
        elif gt in ("Yes", "No"):
            kind, gt = "yesno", gt.lower()
        else:
            kind = "short"
        stem = os.path.basename(r.image["path"]).rsplit(".", 1)[0]
        e = str(r.expected_bias).strip()
        return self.row("vlmsbiased", f"{r.ID}", kind, text, [r.image], gt, None, f"{r.topic}/{r.sub_topic}", group=r.image_path, imgkeys=[stem],
                        expected_bias=e.lower() if kind == "yesno" else e)

    def build(self, bench, ids):
        want = set(ids)
        items = {}
        for sid, item in getattr(self, f"items_{bench}")():
            i = f"{bench}/{sid}"
            if i in want:
                items[i] = (sid, item)
        missing = sorted(want - set(items))
        if missing:
            raise SourceError(f"{bench}: {len(missing)} manifest rows not found in the official files, first {missing[:5]}")
        out = {}
        for i in ids:
            sid, item = items[i]
            out[i] = self.build_charxiv(item, sid) if bench == "charxiv" else getattr(self, f"build_{bench}")(item)
        return out


def canon_gold(s):
    s = s.strip().replace("−", "-")
    if "=" in s:
        s = s.split("=")[-1].strip()
    return s.rstrip(".").strip()


def blank_kind(g):
    if re.fullmatch(r"-?\d+(\.\d+)?", g):
        return "num"
    words = re.findall(r"[A-Za-z]{2,}", g)
    filler = {"row", "column", "col", "and", "the", "th", "st", "nd", "rd"}
    if len([w for w in words if w.lower() not in filler]) > 6:
        return None
    return "short"


def number_instruction(answer):
    base = answer.split(".")
    decimal = None if len(base) == 1 else base[1]
    if decimal is None:
        return "* Your final answer must be an exact integer."
    return f"* Your final answer must be a number with {len(decimal)} decimal places."


def manifest_path(cfg):
    m = cfg["development"]["manifest"]
    return m["file"] if os.path.isabs(m["file"]) else os.path.join(config.CONFIG_DIR, m["file"])


def build_from_sources(sources_dir, images_dir, out_path, cfg=None, scfg=None, check_file_sha=True):
    cfg = cfg or config.load("benchmarks")
    scfg = scfg or config.load("sources")
    mpath = manifest_path(cfg)
    if cfg["development"]["manifest"].get("sha256") and RW.sha256_file(mpath) != cfg["development"]["manifest"]["sha256"]:
        raise SourceError(f"{mpath}: sha256 differs from benchmarks.yaml development.manifest.sha256")
    man = RW.read_jsonl(mpath)
    benches = [b for b in cfg["benchmarks"] if hasattr(Builder, f"items_{b}")]
    extra = sorted({m["bench"] for m in man} - set(benches))
    if extra:
        raise SourceError(f"manifest rows of benchmarks without a loader: {extra}")
    check_files(sources_dir, scfg, benches, check_file_sha)
    B = Builder(sources_dir, images_dir, scfg)
    built = {}
    for b in benches:
        built.update(B.build(b, [m["id"] for m in man if m["bench"] == b]))
    rows, failures, dhash_recomputed = [], [], []
    for m in man:
        r = built[m["id"]]
        if r["img_dhash"] != m["img_dhash"]:
            dhash_recomputed.append(m["id"])
        r["img_dhash"] = list(m["img_dhash"])
        r["pic"] = m["pic"]
        r[RW.POOL_ROLE_FIELD] = m["pool_role"]
        r[RW.ROLE_FIELD] = m["role"]
        e = M.entry(r)
        bad = [k for k in ("img_sha256", "row_sha256") if e[k] != m[k]]
        if bad:
            failures.append(dict(id=m["id"], fields=bad))
        rows.append(r)
    if failures:
        raise SourceError(f"{len(failures)} rows do not match the manifest: " + "; ".join(f"{f['id']} ({', '.join(f['fields'])})" for f in failures[:20]))
    counts = RW.check_counts(rows, cfg)
    sha = RW.write_rows(rows, out_path)
    return dict(development_rows=out_path, sha256=sha, rows=len(rows), counts=counts, manifest=mpath, images_dir=images_dir, dhash_recomputed_differs=dhash_recomputed,
                prompt_sources=scfg.get("prompt_sources"))
