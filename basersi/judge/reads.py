import hashlib
import json
import os
import time

from basersi import config
from basersi.data.rows import ROLE_FIELD, ROLES, SIGNAL_ROLES, RowsError, check_field_names

_BENCH_CFG = config.load("benchmarks")
DEVELOPMENT_SHA256 = _BENCH_CFG["development"]["sha256"]
BENCHES = tuple(_BENCH_CFG["benchmarks"])


class RowError(config.Refused, ValueError):
    pass


def superseded_name(path):
    base = f"{path}.superseded-{time.strftime('%Y%m%d-%H%M%S')}"
    name, n = base, 1
    while os.path.lexists(name):
        n += 1
        name = f"{base}-{n}"
    return name


def load_rows(path):
    raw = open(config.need_file(path, "the development rows file"), "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    lines = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    try:
        check_field_names(lines, path)
    except RowsError as e:
        raise RowError(str(e)) from None
    if sha != DEVELOPMENT_SHA256:
        from basersi.data import manifest
        if not manifest.same_rows(lines, _BENCH_CFG):
            raise RowError(f"{path}: sha256 {sha[:16]}... is not the development row file ({DEVELOPMENT_SHA256[:16]}...) and its rows do not match the shipped row manifest")
    rows = {}
    for r in lines:
        role = r.get(ROLE_FIELD)
        if role not in ROLES:
            raise RowError(f"{path}: row {r.get('id')} has role {role!r}: only show / monitor / judge rows may be loaded")
        if r["bench"] not in BENCHES:
            raise RowError(f"{path}: row {r.get('id')} is from benchmark {r['bench']!r}, not one of {list(BENCHES)}")
        rows[r["id"]] = dict(id=r["id"], bench=r["bench"], role=role, pic=r.get("pic") or r["id"], kind=r.get("kind"),
                             category=r.get("category"))
    return rows


def rows_of(rows, roles):
    roles = (roles,) if isinstance(roles, str) else tuple(roles)
    return sorted(r for r, x in rows.items() if x["role"] in roles)
