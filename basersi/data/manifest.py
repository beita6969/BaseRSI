import hashlib
import json
import os

from basersi.data import rows as RW

LOCAL_FIELDS = ("image_paths",)


def portable(row):
    return {k: v for k, v in row.items() if k not in LOCAL_FIELDS}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def entry(row):
    return dict(id=row["id"], bench=row["bench"], role=row[RW.ROLE_FIELD], pool_role=row.get(RW.POOL_ROLE_FIELD), pic=row.get("pic"),
                img_sha256=list(row.get("img_sha256") or []), img_dhash=list(row.get("img_dhash") or []), row_sha256=digest(portable(row)))


def make(rows_path, out_path):
    if os.path.exists(out_path):
        raise RW.RowsError(f"{out_path} exists: a manifest is never overwritten")
    rows = RW.read_jsonl(rows_path)
    with open(out_path + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(entry(r), ensure_ascii=False) + "\n")
    os.replace(out_path + ".tmp", out_path)
    return dict(manifest=out_path, rows=len(rows), sha256=RW.sha256_file(out_path))


def image_sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def verify(rows_path, manifest_path, check_images=True):
    rows = {r["id"]: r for r in RW.read_jsonl(rows_path)}
    man = {m["id"]: m for m in RW.read_jsonl(manifest_path)}
    out = dict(rows=len(rows), manifest=len(man), missing=sorted(set(man) - set(rows)), extra=sorted(set(rows) - set(man)),
               role=[], images=[], image_files=[], row=[])
    for i in sorted(set(man) & set(rows)):
        m, e = man[i], entry(rows[i])
        for k, key in (("role", "role"), ("images", "img_sha256"), ("row", "row_sha256")):
            if m[key] != e[key]:
                out[k].append(i)
        if check_images:
            paths = rows[i].get("image_paths") or []
            if len(paths) != len(m["img_sha256"]) or any(not os.path.exists(p) or image_sha256(p) != s for p, s in zip(paths, m["img_sha256"])):
                out["image_files"].append(i)
    out["ok"] = not any(out[k] for k in ("missing", "extra", "role", "images", "image_files", "row"))
    return out


def shipped_path(cfg):
    m = cfg["development"].get("manifest") or {}
    if not m.get("file"):
        return None
    from basersi import config
    return m["file"] if os.path.isabs(m["file"]) else os.path.join(config.CONFIG_DIR, m["file"])


def same_rows(rows, cfg):
    path = shipped_path(cfg)
    if not path or not os.path.exists(path):
        return False
    man = {m["id"]: m["row_sha256"] for m in RW.read_jsonl(path)}
    got = {r["id"]: digest(portable(r)) for r in rows}
    return got == man
