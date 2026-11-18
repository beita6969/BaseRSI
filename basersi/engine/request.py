import base64
import io
import math
import mimetypes
import os
import re

from basersi import config, fingerprint

_ENGINE = config.load("engine")
FACTOR = int(_ENGINE["image"]["merged_patch_pixels"])
STOCK_MIN_PX = int(_ENGINE["image"]["stock_min_pixels"])
STOCK_MAX_PX = int(_ENGINE["image"]["stock_max_pixels"])
MAX_ASPECT_RATIO = float(_ENGINE["image"]["max_aspect_ratio"])
OPS_IMAGE_LIMIT = int(_ENGINE["ops"]["image_limit"])
OPS_MAXLEN = int(_ENGINE["ops"]["max_model_len"])
OPS_GEN_ROOM = int(_ENGINE["ops"]["generation_room"])
MIN_OP_TOKENS = int(_ENGINE["ops"]["min_op_tokens"])
MIN_SIDE = float(_ENGINE["ops"]["min_box_side"])

ZOOM_Q = ("Do not answer the question. Name the image region that matters most for answering it: reply with only [i, x0, y0, x1, y1], "
          "where i is the image number (1 for a single image) and x0 y0 x1 y1 are normalised coordinates between 0 and 1.")
CHECK_Q = ("A proposed final answer to this question: {a}. Check it against the image(s) and the question. If it is wrong, correct it. "
           "Give your final answer in \\boxed{{Answer}} format.")


_ROW_IMAGES = {}
_MANIFEST_IMAGES = {}


class ImageRefused(config.Refused, ValueError):
    pass


def manifest_images():
    if not _MANIFEST_IMAGES:
        from basersi.data import manifest
        from basersi.data.rows import read_jsonl
        _MANIFEST_IMAGES.update({m["id"]: list(m["img_sha256"]) for m in read_jsonl(manifest.shipped_path(config.load("benchmarks")))})
    return _MANIFEST_IMAGES


def register_row_images(rows):
    for r in rows:
        paths, shas = list(r.get("image_paths") or []), list(r.get("img_sha256") or [])
        if len(paths) != len(shas):
            raise ImageRefused(f"row {r.get('id')}: {len(paths)} image paths but {len(shas)} images in the row manifest")
        for p, h in zip(paths, shas):
            _ROW_IMAGES.setdefault(os.path.realpath(p), {})[h] = r["id"]


def checked_image(path, row=None):
    real = os.path.realpath(path)
    if row is not None:
        shas = manifest_images().get(row.get("id"))
        if shas is None:
            raise ImageRefused(f"row {row.get('id')!r} is not in the shipped row manifest: its images are not read")
        want = {h: row["id"] for h in shas}
    else:
        want = _ROW_IMAGES.get(real)
        if not want:
            raise ImageRefused(f"image {path} is not an image of a verified development row: only the images of rows checked against the shipped row manifest are read")
    rows = ", ".join(sorted(set(want.values())))
    if not os.path.isfile(real):
        raise ImageRefused(f"row {rows}: image {path} is missing")
    got = fingerprint.file_sha256(real)
    if got not in want:
        raise ImageRefused(f"row {rows}: image {path} is not the row's image (its sha256 {got[:16]}... is not in the row manifest's "
                           f"{', '.join(h[:16] + '...' for h in sorted(want)) or 'empty list'}); point image_paths at the row's own image files")
    return real


def verify_images(rows):
    man = manifest_images()
    for r in rows:
        paths = list(r.get("image_paths") or [])
        shas = man.get(r.get("id"))
        if shas is None:
            raise ImageRefused(f"row {r.get('id')!r} is not in the shipped row manifest: its images are not read")
        if len(paths) != len(shas):
            raise ImageRefused(f"row {r['id']}: {len(paths)} image paths, the row manifest has {len(shas)} images")
        for k, (p, h) in enumerate(zip(paths, shas)):
            if fingerprint.file_sha256(checked_image(p, r)) != h:
                raise ImageRefused(f"row {r['id']}: image {p} is not image {k + 1} of the row in the row manifest (the images are in another order)")
        register_row_images([dict(id=r["id"], image_paths=paths, img_sha256=shas)])


def data_url(path):
    path = checked_image(path)
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def smart_resize(h, w, factor=FACTOR, min_pixels=STOCK_MIN_PX, max_pixels=STOCK_MAX_PX):
    if max(h, w) / min(h, w) > MAX_ASPECT_RATIO:
        raise ValueError(f"aspect ratio > {MAX_ASPECT_RATIO:g}: {h}x{w}")
    hb = round(h / factor) * factor
    wb = round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def view_pixels(min_tokens=None, max_tokens=None):
    mx = max_tokens * FACTOR ** 2 if max_tokens else STOCK_MAX_PX
    mn = min_tokens * FACTOR ** 2 if min_tokens else min(STOCK_MIN_PX, mx)
    return mn, mx


def load_rgb(path, row=None):
    from PIL import Image
    im = Image.open(checked_image(path, row))
    im.load()
    if im.mode == "RGB":
        return im
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def _stock_bounds(min_tokens, max_tokens):
    if max_tokens and max_tokens * FACTOR ** 2 >= STOCK_MAX_PX:
        max_tokens = None
    if min_tokens and min_tokens * FACTOR ** 2 <= STOCK_MIN_PX:
        min_tokens = None
    return min_tokens, max_tokens


def image_url_for(path, min_tokens=None, max_tokens=None):
    from PIL import Image
    min_tokens, max_tokens = _stock_bounds(min_tokens, max_tokens)
    if not min_tokens and not max_tokens:
        with Image.open(checked_image(path)) as im:
            w, h = im.size
        hb, wb = smart_resize(h, w)
        return data_url(path), (hb // FACTOR) * (wb // FACTOR)
    mn, mx = view_pixels(min_tokens, max_tokens)
    im = load_rgb(path)
    w, h = im.size
    hb, wb = smart_resize(h, w, min_pixels=mn, max_pixels=mx)
    hs, ws = smart_resize(hb, wb)
    if (hb, wb) != (h, w):
        im = im.resize((wb, hb), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), (hs // FACTOR) * (ws // FACTOR)


def image_tokens_for(path, min_tokens=None, max_tokens=None):
    from PIL import Image
    with Image.open(checked_image(path)) as im:
        w, h = im.size
    min_tokens, max_tokens = _stock_bounds(min_tokens, max_tokens)
    mn, mx = view_pixels(min_tokens, max_tokens) if (min_tokens or max_tokens) else (STOCK_MIN_PX, STOCK_MAX_PX)
    hb, wb = smart_resize(h, w, min_pixels=mn, max_pixels=mx)
    return (hb // FACTOR) * (wb // FACTOR)


def content_for(prompt, image_paths, min_tokens=None, max_tokens=None, prefix=None):
    imgs, ntok = [], []
    for p in image_paths:
        u, n = image_url_for(p, min_tokens, max_tokens)
        imgs.append({"type": "image_url", "image_url": {"url": u}})
        ntok.append(n)
    parts = prompt.split("<image>")
    if len(parts) - 1 == len(imgs) and imgs:
        content = []
        if prefix:
            content.append({"type": "text", "text": prefix.rstrip() + "\n\n"})
        for k, txt in enumerate(parts):
            if txt.strip():
                content.append({"type": "text", "text": txt.strip()})
            if k < len(imgs):
                content.append(imgs[k])
        return content, ntok
    pre = [{"type": "text", "text": prefix.rstrip() + "\n\n"}] if prefix else []
    return imgs + pre + [{"type": "text", "text": prompt}], ntok


def chat_messages(messages):
    out, ntok = [], []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if p.get("type") == "image":
                    u, n = image_url_for(p["path"], p.get("min_image_tokens"), p.get("max_image_tokens"))
                    parts.append({"type": "image_url", "image_url": {"url": u}})
                    ntok.append(n)
                elif p.get("type") == "text":
                    parts.append({"type": "text", "text": p["text"]})
                else:
                    parts.append(p)
            m = dict(m, content=parts)
        out.append(m)
    return out, ntok


def parse_box(text, n_images):
    m = re.search(r"\[\s*(\d+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", text or "")
    if not m:
        return None
    i = int(m.group(1))
    x0, y0, x1, y1 = (min(1.0, max(0.0, float(m.group(k)))) for k in range(2, 6))
    if not 1 <= i <= n_images:
        return None
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    def widen(lo, hi):
        if hi - lo >= MIN_SIDE:
            return lo, hi
        c = (lo + hi) / 2
        lo, hi = c - MIN_SIDE / 2, c + MIN_SIDE / 2
        if lo < 0:
            lo, hi = 0.0, MIN_SIDE
        if hi > 1:
            lo, hi = 1 - MIN_SIDE, 1.0
        return lo, hi
    x0, x1 = widen(x0, x1)
    y0, y1 = widen(y0, y1)
    return [i, round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def upscaled_part(im, tokens):
    from PIL import Image
    T = max(MIN_OP_TOKENS, int(tokens))
    w, h = im.size
    hb, wb = smart_resize(h, w, min_pixels=T * FACTOR ** 2, max_pixels=T * FACTOR ** 2)
    im = im.resize((wb, hb), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=1)
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}, (hb // FACTOR) * (wb // FACTOR)


def crop_part(path, box, tokens):
    im = load_rgb(path)
    w, h = im.size
    _, x0, y0, x1, y1 = box
    c = im.crop((int(x0 * w), int(y0 * h), max(int(x0 * w) + 1, int(round(x1 * w))), max(int(y0 * h) + 1, int(round(y1 * h)))))
    return upscaled_part(c, tokens)


def tile_parts(path, n, tokens):
    im = load_rgb(path)
    w, h = im.size
    out = []
    for r in range(n):
        for c in range(n):
            t = im.crop((c * w // n, r * h // n, (c + 1) * w // n, (r + 1) * h // n))
            out.append(upscaled_part(t, tokens))
    return out


def insert_after_last_image(content, parts):
    k = max((i for i, p in enumerate(content) if p.get("type") == "image_url"), default=-1)
    return content[:k + 1] + list(parts) + content[k + 1:]


def ops_need(row, ops, settings=None):
    st = settings or {}
    paths = row.get("image_paths") or []
    toks = [image_tokens_for(p, st.get("min_image_tokens"), st.get("max_image_tokens")) for p in paths]
    n_img, est = len(paths), sum(toks)
    for o in ops:
        if o["op"] == "zoom" and paths:
            n_img += 1
            est += max(MIN_OP_TOKENS, max(toks))
        elif o["op"] == "tiles":
            k = int(o.get("n", 2))
            n_img += k * k * len(paths)
            est += sum(k * k * max(MIN_OP_TOKENS, t) for t in toks)
    est += len(row.get("prompt") or "") // 3 + 64
    return n_img, est


def ops_skip_reason(row, ops, settings=None):
    n, est = ops_need(row, ops, settings)
    if n > OPS_IMAGE_LIMIT:
        return f"image_limit: {n} images > {OPS_IMAGE_LIMIT}"
    if est > OPS_MAXLEN - OPS_GEN_ROOM:
        return f"context: about {est} prompt tokens > {OPS_MAXLEN - OPS_GEN_ROOM}"
    return None
