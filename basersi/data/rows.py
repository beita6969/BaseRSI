import collections
import hashlib
import json
import os
import re

import numpy as np

from basersi import config

ROLE_FIELD = "role"
POOL_ROLE_FIELD = "pool_role"
ROLES = ("show", "monitor", "judge")
VALIDATION_ROLE = "judge"
SIGNAL_ROLES = tuple(r for r in ROLES if r != VALIDATION_ROLE)
PROPOSER_ROLES = ("show",)


class RowsError(config.Refused, ValueError):
    pass


def check_field_names(rows, path, required=(ROLE_FIELD,)):
    for field in required:
        lacking = [r for r in rows if field not in r]
        if lacking:
            raise RowsError(f"{path}: {len(lacking)} of {len(rows)} rows have no field {field!r} (fields of the first: {sorted(lacking[0])}); "
                            f"rows of this release name their role field {ROLE_FIELD!r}")


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def read_jsonl(path):
    return [json.loads(l) for l in open(config.need_file(path, "the file"), encoding="utf-8") if l.strip()]


def check_counts(rows, cfg=None):
    cfg = cfg or config.load("benchmarks")
    c = collections.Counter((r["bench"], r[ROLE_FIELD]) for r in rows)
    per = cfg.get("rows_per_benchmark")
    if per is not None and sum(cfg["roles"].values()) != int(per):
        raise RowsError(f"benchmarks.yaml: roles {cfg['roles']} add up to {sum(cfg['roles'].values())}, not rows_per_benchmark {per}")
    bad = [(b, role, c[(b, role)], n) for b in cfg["benchmarks"] for role, n in cfg["roles"].items() if c[(b, role)] != n]
    if bad:
        raise RowsError(f"role counts differ from benchmarks.yaml: {bad}")
    extra = sorted({r["bench"] for r in rows} - set(cfg["benchmarks"]))
    if extra:
        raise RowsError(f"rows of benchmarks outside benchmarks.yaml: {extra}")
    n = cfg["development"]["rows"]
    if len(rows) != n or len({r["id"] for r in rows}) != n:
        raise RowsError(f"{len(rows)} rows ({len({r['id'] for r in rows})} distinct ids), expected {n}")
    return {f"{b}|{role}": k for (b, role), k in sorted(c.items())}


def write_rows(rows, path):
    if os.path.exists(path):
        from basersi.fingerprint import move_commands
        raise RowsError(f"{path} exists: development rows are never overwritten; write to another --out, or move it away first: {move_commands([path])}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    return sha256_file(path)


def id_stem(row, cfg):
    pat = (cfg.get("id_stem_patterns") or {}).get(row["bench"])
    return re.sub(pat, "", row["id"]) if pat else None


def row_keys(row, cfg):
    out = [("id", row["id"])] + [("sha", s) for s in row.get("img_sha256") or []]
    if row.get("group") is not None:
        out.append(("group", f"{row['bench']}|{row['group']}"))
    stem = id_stem(row, cfg)
    if stem:
        out.append(("stem", stem))
    if row.get("pic") is not None:
        out.append(("pic", f"{row['bench']}|{row['pic']}"))
    return out


def shared_keys(A, B, cfg):
    ka = collections.defaultdict(set)
    for r in A:
        for k in row_keys(r, cfg):
            ka[k].add(r["id"])
    hit, seen = collections.Counter(), set()
    for r in B:
        for k in row_keys(r, cfg):
            if k in ka and (k, r["id"]) not in seen:
                seen.add((k, r["id"]))
                hit[k[0]] += 1
    return {t: hit.get(t, 0) for t in ("id", "sha", "group", "stem", "pic")}


def dhash_matrix(rows):
    D, W = [], []
    for r in rows:
        for d in r.get("img_dhash") or []:
            if d:
                D.append([int(d[k:k + 16], 16) for k in range(0, 64, 16)])
                W.append(r["id"])
    return np.array(D, dtype=np.uint64), W


def dhash_overlap(A, B, threshold):
    a, wa = dhash_matrix(A)
    b, wb = dhash_matrix(B)
    n, mind = 0, 256
    for s in range(0, len(a), 256):
        hd = np.bitwise_count(a[s:s + 256][:, None, :] ^ b[None, :, :]).sum(-1)
        for p, q in zip(*np.nonzero(hd <= threshold)):
            if wa[s + p] != wb[q]:
                n += 1
        mask = np.array([[wa[s + p] != wb[q] for q in range(len(wb))] for p in range(hd.shape[0])])
        mind = min(mind, int(hd[mask].min()) if mask.any() else 256)
    return dict(pairs_within_threshold=n, min_hamming=mind)


def overlap_judge_signal(rows, cfg=None):
    cfg = cfg or config.load("benchmarks")
    signal = set(SIGNAL_ROLES)
    J = [r for r in rows if r[ROLE_FIELD] == VALIDATION_ROLE]
    S = [r for r in rows if r[ROLE_FIELD] in signal]
    return dict(rows=[len(J), len(S)], shared_keys=shared_keys(J, S, cfg), dhash=dhash_overlap(J, S, cfg["overlap_max_dhash"]))


def validate(path, cfg=None):
    cfg = cfg or config.load("benchmarks")
    check_field_names(read_jsonl(path), path)
    sha = sha256_file(path)
    if sha != cfg["development"]["sha256"]:
        from basersi.data import manifest
        if not manifest.same_rows(read_jsonl(path), cfg):
            raise RowsError(f"{path}: sha256 {sha[:16]}... is not the development row file ({cfg['development']['sha256'][:16]}...) and its rows do not match the shipped row manifest")
    rows = read_jsonl(path)
    for r in rows:
        if r.get(ROLE_FIELD) not in ROLES:
            raise RowsError(f"row {r.get('id')}: role {r.get(ROLE_FIELD)!r}")
    counts = check_counts(rows, cfg)
    ov = overlap_judge_signal(rows, cfg)
    if any(ov["shared_keys"].values()) or ov["dhash"]["pairs_within_threshold"]:
        raise RowsError(f"JUDGE and signal rows overlap: {ov}")
    return dict(development_rows=path, sha256=sha, counts=counts, overlap_judge_signal=ov)
