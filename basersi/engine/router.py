import hashlib
import math

from basersi import config
from basersi.engine.request import content_for

_ROUTER = config.load("engine")["router"]
ROUTE_PROMPT = "Condition: {c}.\nDo not solve the question above. Does this condition hold for it? Reply with exactly one word: yes or no."
ROUTER_VERSION = _ROUTER["version"]
ROUTED_IF_P_AT_LEAST = float(_ROUTER["routed_if_p_at_least"])
ROUTE_KEY_PREFIX = "rt1"


def condition_hash(condition):
    return hashlib.sha256(condition.encode()).hexdigest()[:16]


def route_key(cond_hash, row_id):
    return f"{ROUTE_KEY_PREFIX}|{cond_hash}|{row_id}"


def route_body(row, condition, view=None):
    v = view or {}
    content, nt = content_for(row["prompt"], row.get("image_paths") or [], v.get("min_image_tokens"), v.get("max_image_tokens"))
    content = list(content) + [{"type": "text", "text": ROUTE_PROMPT.format(c=condition.strip().rstrip("."))}]
    body = dict(messages=[{"role": "user", "content": content}], max_tokens=int(_ROUTER["max_tokens"]), temperature=0.0, logprobs=True,
                top_logprobs=int(_ROUTER["top_logprobs"]), chat_template_kwargs={"enable_thinking": False}, structured_outputs={"choice": ["yes", "no"]})
    return body, nt


def yes_probability(choice, content):
    py = pn = 0.0
    try:
        for t in choice["logprobs"]["content"][0]["top_logprobs"]:
            w = t["token"].strip().lower().strip(".,!")
            if w == "yes":
                py += math.exp(t["logprob"])
            elif w == "no":
                pn += math.exp(t["logprob"])
    except (KeyError, IndexError, TypeError):
        pass
    if py + pn > 0:
        p, src = py / (py + pn), "logprob"
    else:
        p, src = (1.0 if content.strip().lower() == "yes" else 0.0), "choice"
    return dict(yes=p >= ROUTED_IF_P_AT_LEAST, p=p, py=py, pn=pn, p_src=src)


def routed(answer):
    p = answer.get("p")
    return bool(p >= ROUTED_IF_P_AT_LEAST) if isinstance(p, (int, float)) else bool(answer.get("yes"))
