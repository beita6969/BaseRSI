import argparse
import json
import os

from basersi import config as C
from basersi.data import manifest as M
from basersi.data import rows as RW

HELP = "development rows: build them from the official benchmark files and the shipped row manifest, check a file, or write and verify a row manifest"


def main(argv=None):
    form = dict(epilog=C.EXIT_CODES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap = argparse.ArgumentParser(prog="basersi data", description=HELP, **form)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="rebuild the development rows from the official benchmark files (--from-sources) and check every digest against the shipped manifest", **form)
    b.add_argument("--from-sources", dest="sources", required=True, help="folder with the official benchmark files at the revisions of sources.yaml (<repo with / as __>/<file>); the rows of the shipped manifest are rebuilt and every digest is checked")
    b.add_argument("--images", default=None, help="with --from-sources: folder for the image files (default: <out folder>/images)")
    b.add_argument("--no-file-sha", action="store_true", help="with --from-sources: skip the size and sha256 check of the official files (the row digests are still checked)")
    b.add_argument("--out", required=True, help="development rows file to write (never overwritten)")
    v = sub.add_parser("check", help="check a development rows file against the shipped sha256 and row manifest, its role counts and JUDGE / signal overlap", **form)
    v.add_argument("--rows", required=True, help="development rows file")
    m = sub.add_parser("manifest", help="write the row manifest of a development rows file (id, benchmark, role, picture group, image sha256 and perceptual hash, row digest)", **form)
    m.add_argument("--rows", required=True, help="development rows file")
    m.add_argument("--out", required=True, help="manifest file to write (never overwritten)")
    w = sub.add_parser("verify", help="compare a development rows file with a row manifest: ids, roles, row digests and the sha256 of the image files (exit 2 when they differ)", **form)
    w.add_argument("--rows", required=True, help="development rows file")
    w.add_argument("--manifest", required=True, help="row manifest")
    w.add_argument("--no-images", action="store_true", help="do not read the image files")
    for p in (b, v, m, w):
        p.add_argument("--config", default=None, help="yaml file merged over basersi/config/benchmarks.yaml for this command's own checks only (role counts, rows, sha256);"
                       " the judge, the loop and the pipeline always use the shipped benchmarks.yaml (benchmark list, roles, development sha256)")
    a = ap.parse_args(argv)
    cfg = C.load("benchmarks", a.config)
    if a.cmd == "build":
        from basersi.data import sources as SR
        images = a.images or os.path.join(os.path.dirname(os.path.abspath(a.out)), "images")
        res = SR.build_from_sources(a.sources, images, a.out, cfg, check_file_sha=not a.no_file_sha)
        res["check"] = RW.validate(a.out, cfg)
    elif a.cmd == "manifest":
        res = M.make(a.rows, a.out)
    elif a.cmd == "verify":
        res = M.verify(a.rows, a.manifest, check_images=not a.no_images)
        print(json.dumps(res, indent=1))
        if not res["ok"]:
            raise RW.RowsError(f"{a.rows} differs from the manifest {a.manifest}: see the report above")
        return 0
    else:
        res = RW.validate(a.rows, cfg)
    print(json.dumps(res, indent=1))
    return 0
