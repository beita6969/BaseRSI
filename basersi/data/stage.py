import os

from basersi import config as C
from basersi.data import rows as RW


def run(config):
    section = dict(config.get("data") or {})
    cfg = C.load("benchmarks")
    path = section.get("development_rows") or os.path.join(config["data_dir"], cfg["development"]["file"])
    built = None
    if not os.path.exists(path):
        if not section.get("sources"):
            raise C.MissingInput(f"{path} does not exist and data.sources does not name the folder of the official benchmark files to build it from")
        from basersi.data import sources as SR
        images = section.get("images_dir") or os.path.join(os.path.dirname(os.path.abspath(path)), "images")
        built = SR.build_from_sources(section["sources"], images, path, cfg)
        built["excluded_source"] = "manifest"
    try:
        res = RW.validate(path, cfg)
    except RW.RowsError as e:
        from basersi.fingerprint import move_commands
        how = ("the official benchmark files (data.sources) and the shipped row manifest" if section.get("sources")
               else "the official benchmark files and the shipped row manifest once data.sources names their folder")
        raise RW.RowsError(f"{e}\n{path} is not a valid development rows file; move it away and run the data step again, which rebuilds it from {how}:\n  "
                           f"{move_commands([path])}") from None
    res["built"] = built is not None
    if built:
        res["excluded_source"] = built["excluded_source"]
    return res
