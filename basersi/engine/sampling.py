import hashlib
import zlib

from basersi import config

_ENGINE = config.load("engine")
SAMPLING = {k: float(v) if k != "top_k" else int(v) for k, v in _ENGINE["sampling"].items()}
EFFORT = _ENGINE["reasoning_effort"]
READ_MAX = int(_ENGINE["read_max_tokens"])
CHAT_MAX = int(_ENGINE["chat_max_tokens"])
CHAT_TEMP = float(_ENGINE["chat_temperature"])
CRN_SEED = int(_ENGINE["crn_seed_base"])
CRN_GENERATION = int(_ENGINE["crn_seed_generation"])
CRN_MODULUS = int(_ENGINE["crn_seed_modulus"])


def crn_seed(row, sample):
    return zlib.crc32(f"{CRN_SEED}|{CRN_GENERATION}|{row}".encode()) % CRN_MODULUS + int(sample)


def cache_salt(request):
    return hashlib.sha256(("s|k|" + request["key"]).encode()).hexdigest()[:32]


def thinking_kwargs():
    return {"reasoning_effort": EFFORT, "enable_thinking": True}


def read_body(row, settings=None, prefix=None, sample=0, seed=None, max_tokens=None):
    from basersi.engine.request import content_for
    st = settings or {}
    content, nt = content_for(row["prompt"], row.get("image_paths") or [], st.get("min_image_tokens"), st.get("max_image_tokens"), prefix)
    body = dict(messages=[{"role": "user", "content": content}], max_tokens=int(max_tokens or READ_MAX), chat_template_kwargs=thinking_kwargs(), **SAMPLING)
    if st.get("temperature") is not None:
        body["temperature"] = float(st["temperature"])
    body["seed"] = int(seed) if seed is not None else crn_seed(row["id"], sample)
    return body, nt
