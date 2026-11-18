import collections
import json
import random
import re

from basersi import config
from basersi.data.rows import PROPOSER_ROLES, ROLE_FIELD, check_field_names


def configure(p):
    global COND_MAX, PROC_MAX, REASON_TAIL, CONTENT_TAIL, SAMPLES_SHOWN, MIN_TEXT, ANCHOR, CATEGORY_MIN_WORDS, TAIL_SCAN
    COND_MAX, PROC_MAX = int(p["condition_max_chars"]), int(p["procedure_max_chars"])
    REASON_TAIL, CONTENT_TAIL = int(p["reasoning_tail_chars"]), int(p["content_tail_chars"])
    SAMPLES_SHOWN = int(p["samples_shown"])
    MIN_TEXT = int(p["guard_min_text_chars"])
    ANCHOR = MIN_TEXT
    CATEGORY_MIN_WORDS = int(p["category_min_words"])
    TAIL_SCAN = int(p["tail_scan_chars"])


configure(config.load("loop")["proposer"])
CAT_SEP = re.compile(r"[\s_/]+")

OPS = {"zoom": {}, "tiles": {"n": [2, 3]}, "check": {}}
OPS_ORDER = ("zoom", "tiles", "check")
OPS_IMAGE_LIMIT = int(config.load("engine")["ops"]["image_limit"])


RULE_NOTE = ("How entries are judged: an entry is kept in the general system only if it helps on benchmarks other than the one its "
             "failures came from; otherwise it can only enter that benchmark's own layer.")


class LeakError(AssertionError):
    pass


def load_rows(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                if r["id"] in out:
                    raise config.Refused(f"duplicate row id {r['id']!r} in {path}")
                out[r["id"]] = r
    check_field_names(list(out.values()), path)
    return out


def _sidx(s):
    if isinstance(s.get("sample"), int):
        return s["sample"]
    try:
        return int(str(s.get("key", "")).rsplit("|", 1)[-1])
    except ValueError:
        return 0


def _samples(parent, rid):
    vs = [s for s in (parent.get(rid) or []) if isinstance(s, dict) and not s.get("error") and s.get("ok") is not None]
    return sorted(vs, key=_sidx)


def _is_right(vs):
    return 2 * sum(bool(s["ok"]) for s in vs) > len(vs)


def cards_mixed(rows, parent, k, seed, n_wrong=8, n_right=4):
    rng = random.Random(f"{seed}:{k}")
    pools = {False: collections.defaultdict(list), True: collections.defaultdict(list)}
    for rid in sorted(rows):
        r = rows[rid]
        if r.get(ROLE_FIELD) not in PROPOSER_ROLES:
            continue
        vs = _samples(parent, rid)
        if vs:
            pools[_is_right(vs)][r["bench"]].append(rid)
    benches = sorted(set(pools[False]) | set(pools[True]))
    if not benches:
        return []
    rng.shuffle(benches)
    left = {ok: {b: rng.sample(v, len(v)) for b, v in sorted(pools[ok].items())} for ok in (False, True)}
    pos = [0]

    def take(ok, n):
        got = []
        while len(got) < n and any(left[ok].get(b) for b in benches):
            b = benches[pos[0] % len(benches)]
            pos[0] += 1
            if left[ok].get(b):
                got.append(left[ok][b].pop())
        return got

    wrong = take(False, n_wrong)
    want_right = n_right + (n_wrong - len(wrong))
    right = take(True, want_right)
    if len(right) < want_right:
        wrong += take(False, want_right - len(right))
    out = wrong + right
    rng.shuffle(out)
    return out


def source_pools(rows, parent, source):
    pools = {False: [], True: []}
    for rid in sorted(rows):
        r = rows[rid]
        if r.get(ROLE_FIELD) in PROPOSER_ROLES and r.get("bench") == source:
            vs = _samples(parent, rid)
            if vs:
                pools[_is_right(vs)].append(rid)
    return pools[False], pools[True]


def cards_source(rows, parent, source, i, seed, n_wrong=8, n_right=4):
    out = []
    for ok, (pool, n) in enumerate(zip(source_pools(rows, parent, source), (n_wrong, n_right))):
        P = list(pool)
        random.Random(f"{seed}:{source}:{ok}").shuffle(P)
        m = min(n, len(P))
        out += [P[(i * m + t) % len(P)] for t in range(m)]
    random.Random(f"{seed}:{source}:slot{i}").shuffle(out)
    return out


def _ws(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def is_always(c):
    return isinstance(c, str) and c.strip().strip(".\"'").lower() == "always"


def _same(v, x):
    if v is None or x is None:
        return v is None and x is None
    if isinstance(v, bool) or isinstance(x, bool) or not isinstance(v, (int, float)) or not isinstance(x, (int, float)):
        return False
    return float(v) == float(x)


def _in_menu(value, values):
    for x in values:
        if isinstance(x, str) or isinstance(value, str):
            if isinstance(x, str) and isinstance(value, str) and x == value:
                return True
        elif isinstance(x, bool) or isinstance(value, bool):
            if isinstance(x, bool) and isinstance(value, bool) and x == value:
                return True
        elif _same(value, x):
            return True
    return False


def _tip_view(tip, menu):
    base = {k: menu[k]["stock"] for k in menu}
    tip = tip or {}
    base.update({k: v for k, v in (tip.get("settings") or {}).items() if k in menu})
    entries = list(tip.get("entries") or [])
    for e in entries:
        if is_always(e.get("condition")):
            base.update({k: v for k, v in (e.get("settings") or {}).items() if k in menu})
    return base, entries


def _val(v):
    return json.dumps(v)


def entry_key(e):
    c = e.get("condition")
    c = "always" if is_always(c) else _ws(c or "").lower()
    p = e.get("procedure")
    p = _ws(p) if isinstance(p, str) and p.strip() else None
    s = tuple(sorted((str(k), json.dumps(v)) for k, v in (e.get("settings") or {}).items()))
    o = _ops_key(e.get("ops"))
    return (c, p, s, o) if o else (c, p, s)


def _op_rank(o):
    name = o.get("op") if isinstance(o, dict) else None
    return OPS_ORDER.index(name) if name in OPS_ORDER else len(OPS_ORDER)


def _norm_op(o):
    out = {}
    for k, v in o.items():
        k = str(k).strip().lower()
        if k == "op":
            v = v.strip().lower() if isinstance(v, str) else v
        else:
            v = _num(v)
            if isinstance(v, float) and v.is_integer():
                v = int(v)
        out[k] = v
    return out


def canon_ops(ops):
    if not ops:
        return []
    return sorted((_norm_op(o) if isinstance(o, dict) else o for o in ops), key=_op_rank)


def _ops_key(ops):
    if not ops:
        return ()
    if not isinstance(ops, list):
        return (json.dumps(ops, sort_keys=True, default=str),)
    return tuple(json.dumps(o, sort_keys=True, default=str) for o in canon_ops(ops))


def merge_ops(lists):
    by = {}
    for ops in lists:
        for o in canon_ops(ops or []):
            if isinstance(o, dict) and o.get("op") in OPS:
                by[o["op"]] = dict(o)
    return [by[k] for k in OPS_ORDER if k in by]


def _ops_text(ops):
    return ", ".join(json.dumps(o) for o in canon_ops(ops)) if ops else "none"


SYSTEM = """You write one change to the harness of a vision-language model, called the executor.

The executor answers one question at a time. It receives the question's images and text, reasons, and then writes a final answer, which is graded against a gold answer.

The harness is an ordered list of entries. Each entry has a condition and a payload.
- The condition is either "always" or a sentence in words. For each question, a router reads that question's images and text and answers yes or no to the condition. "always" holds on every question.
- The payload is a procedure text, setting changes, harness ops, or a combination. On a question where the condition holds, the procedure is added to the executor's prompt right before the question text, the settings are used for that question, and the ops are applied when the executor answers that question. On a question where the condition does not hold, the entry has no effect.

You write exactly one new entry. It is added to the current harness, called the tip, and judged by its paired gain over the tip on held-out questions that you never see. On each held-out question the tip with your entry and the tip alone both answer, and the gain is the difference in accuracy. Only questions on which your condition holds are read; on all other questions the two are identical and the difference is zero. The gain is scored on the benchmarks other than the one the questions shown to you come from; when they come from several benchmarks, it is scored on all of them. The entry joins the general harness only if its gain across the scored benchmarks passes a significance test for a general gain, corrected for every entry judged in the same generation, it is scored on enough benchmarks, and it is significantly negative on none of them.

The router decides the condition separately for each question, from that question's images and text alone, and the same router also runs on questions from benchmarks that are not shown here.

You see the questions as text only; the router and the executor see the images."""

ANSWER_FORMAT = """## Answer format
Reason first. Then end with exactly one json block:
```json
{"why": "...", "condition": "always" | "<words>", "procedure": "<text>" | null, "settings": {"<knob>": value, ...} | {}, "ops": [{"op": "<op>", ...}, ...] | []}
```"""


def _tail(t, n):
    t = _ws(t or "")
    if len(t) <= n:
        return t
    t = t[-n:]
    sp = t.find(" ")
    return "..." + (t[sp + 1:] if 0 <= sp < 30 else t)


def _gold_lines(r):
    opts, ans = list(r.get("options") or []), str(r.get("answer", "")).strip()
    if r.get("kind") == "mc" and opts:
        letters = [chr(65 + i) for i in range(len(opts))]
        L = ["Options:"] + [f"({a}) {o}" + ("   <- gold" if a == ans else "") for a, o in zip(letters, opts)]
        if ans not in letters:
            L.append(f"Gold answer: {ans}")
        return L
    return [f"Gold answer: {ans}"]


def _pick(vs, wrong_row, n=None):
    n = SAMPLES_SHOWN if n is None else n
    first = next((s for s in vs if bool(s["ok"]) != wrong_row), None) or vs[0]
    rest = [s for s in vs if s is not first]
    other = next((s for s in rest if bool(s["ok"]) != bool(first["ok"])), None)
    out = [first] + ([other] if other is not None else [])
    out += [s for s in rest if all(s is not x for x in out)]
    return out[:n]


def _sample_lines(j, s):
    content = (s.get("content") or "").strip()
    if content:
        said = _tail(content, CONTENT_TAIL)
    else:
        said = "(none" + (", the sample reached the token limit" if s.get("finish") == "length" else "") + ")"
    reason = _tail(s.get("reasoning") or "", REASON_TAIL) or "(empty)"
    return [f"- Sample {j} ({'right' if s['ok'] else 'wrong'})", f"  End of reasoning: {reason}", f"  Final text: {said}"]


def _tip_block(tip, menu):
    base, entries = _tip_view(tip, menu)
    L = ["## The tip", "Settings on every question (an entry whose condition holds may change them for that question):"]
    L += [f"- {k}: {_val(base[k])}" + (" (stock)" if _same(base[k], menu[k]["stock"]) else "") for k in menu]
    if not entries:
        L.append("Entries: none (the tip is the executor as it is).")
    else:
        L.append(f"Entries ({len(entries)}, in order; where several conditions hold on one question, their procedures are joined in this order"
                 f" and their ops are applied together):")
        for j, e in enumerate(entries, 1):
            st = e.get("settings") or {}
            L += [f"Entry {j}", f"- condition: {e.get('condition')}",
                  f"- procedure: {e.get('procedure') if e.get('procedure') else 'none'}",
                  "- settings: " + (", ".join(f"{k} = {_val(v)}" for k, v in st.items()) if st else "none"),
                  f"- ops: {_ops_text(e.get('ops'))}"]
    return "\n".join(L)


def _card_block(rows, parent, card_ids):
    marks = []
    for rid in card_ids:
        vs = _samples(parent, rid)
        marks.append((rid, vs, _is_right(vs)))
    n_w = sum(not ok for _, _, ok in marks)
    ns = sorted({len(vs) for _, vs, _ in marks})
    times = f"{ns[0]} times" if len(ns) == 1 else "several times"
    L = [f"## Shown questions ({len(marks)}: {n_w} currently wrong, {len(marks) - n_w} currently right)",
         f"The tip answered each question {times} (its samples). A question is currently right when more than half of "
         f"its samples are right, and currently wrong otherwise. For up to {SAMPLES_SHOWN} sample{'' if SAMPLES_SHOWN == 1 else 's'} of each question, the "
         f"end of the executor's reasoning and its final text are shown."]
    for q, (rid, vs, ok) in enumerate(marks, 1):
        r = rows[rid]
        L += ["", f"### Question {q}: currently {'right' if ok else 'wrong'} ({sum(bool(s['ok']) for s in vs)} of {len(vs)} samples right)"]
        L += ["Question text:", str(r["prompt"]).strip()] + _gold_lines(r)
        for j, s in enumerate(_pick(vs, not ok, SAMPLES_SHOWN), 1):
            L += _sample_lines(j, s)
    return "\n".join(L)


def _affordance_block(menu):
    L = ["## What an entry can contain",
         f'- condition: "always", or one sentence of at most {COND_MAX} characters that can be answered yes or no from one question\'s images and text.',
         f"- procedure: a text of at most {PROC_MAX} characters, or null.",
         "- settings: an object with zero or more of these keys, each set to one of its listed values:"]
    for k, d in menu.items():
        what = str(d.get("what", "")).strip()
        L.append(f"  - {k}: one of {', '.join(_val(v) for v in d['values'])} (stock {_val(d['stock'])}). {what[:1].upper() + what[1:]}.")
    L += ["- ops: a list of harness ops, each op at most once, or []. Whatever their order in the list, they are applied in the order "
          "zoom, tiles, answer, check:",
          '  - {"op": "zoom"}: a first short pass names one image region, then the question is answered with the original images plus '
          "that region cropped and enlarged as an extra image.",
          '  - {"op": "tiles", "n": 2 | 3}: each image is also given as n x n enlarged tiles.',
          '  - {"op": "check"}: after answering, one more pass shows the question with the proposed answer and asks to confirm or correct '
          "it, and the corrected answer is graded.",
          f"- On a question where the images with the zoom crop and the tiles would number more than {OPS_IMAGE_LIMIT}, or the prompt "
          "would not fit the context, the ops are not applied and the entry has no effect on that question.",
          "- Equal compute in judging: when an entry's generated tokens per question (all passes counted) exceed the tip's, its gain is "
          "measured against the accuracy the tip reaches at the same token cost by answering the question several times and voting, "
          "instead of the tip's mean.",
          "- The entry needs a procedure, settings, ops, or a combination, and must differ from the tip.",
          "- An entry is rejected without being run if its condition or procedure gives an option letter as an answer, "
          "states a fixed answer value for a kind of object or scene, names a dataset, a task category or a question id, "
          "or tells the executor to think longer or shorter."]
    return "\n".join(L)


def _check_card(rows, parent, card_ids):
    bad = [i for i in card_ids if (rows.get(i) or {}).get(ROLE_FIELD) not in PROPOSER_ROLES]
    if bad:
        raise ValueError(f"card rows must be SHOW rows only: {len(bad)} others")
    miss = [i for i in card_ids if not _samples(parent, i)]
    if miss:
        raise ValueError(f"card rows without a read parent sample: {len(miss)}")


def card_prompt(rows, parent, card_ids, tip, menu):
    _check_card(rows, parent, card_ids)
    user = _tip_block(tip, menu) + "\n\n" + _card_block(rows, parent, card_ids) + "\n\n" + "\n\n".join([_affordance_block(menu), ANSWER_FORMAT])
    guard(SYSTEM, user, rows)
    return SYSTEM, user


FENCE = re.compile(r"```[ \t]*([A-Za-z]*)[ \t]*\n?(.*?)```", re.S)


def _tail_object(text):
    t = text.rstrip()
    if not t.endswith("}"):
        return None
    lo = max(0, len(t) - TAIL_SCAN)
    dec = json.JSONDecoder()
    for i in (m.start() for m in re.finditer(r"\{", t[lo:])):
        try:
            obj, end = dec.raw_decode(t, lo + i)
        except ValueError:
            continue
        if end == len(t) and isinstance(obj, dict):
            return t[lo + i:]
    return None


def _num(v):
    if isinstance(v, str):
        x = v.strip().lower()
        if x in ("null", "none"):
            return None
        if re.fullmatch(r"-?\d+", x):
            return int(x)
        if re.fullmatch(r"-?\d*\.\d+(e-?\d+)?|-?\d+e-?\d+", x):
            return float(x)
    return v


def extract_json(text):
    if not isinstance(text, str) or not text.strip():
        return None, "empty: no text"
    blocks = [b for tag, b in FENCE.findall(text) if tag.lower() == "json" or (not tag and b.strip().startswith("{"))]
    src = blocks[-1] if blocks else _tail_object(text)
    if src is None:
        return None, "no_json: no ```json block and no JSON object at the end"
    try:
        obj = json.loads(src)
    except json.JSONDecodeError as e:
        return None, f"bad_json: {e.msg} at line {e.lineno} column {e.colno}"
    if not isinstance(obj, dict):
        return None, f"not_object: the json is a {type(obj).__name__}"
    return obj, None


def parse(text):
    obj, err = extract_json(text)
    if err:
        return None, err
    if "condition" not in obj:
        return None, "field: condition missing"
    c = obj["condition"]
    if not isinstance(c, str):
        return None, "field: condition must be a string"
    c = "always" if is_always(c) else _ws(c)
    p = obj.get("procedure")
    if isinstance(p, str):
        p = p.strip()
        if p.lower() in ("", "null", "none"):
            p = None
    elif p is not None:
        return None, "field: procedure must be a string or null"
    s = obj.get("settings")
    if s is None:
        s = {}
    if not isinstance(s, dict):
        return None, "field: settings must be an object"
    o = obj.get("ops")
    if o is None:
        o = []
    if not isinstance(o, list) or not all(isinstance(x, dict) for x in o):
        return None, "field: ops must be a list of objects"
    why = obj.get("why")
    cand = dict(why=_ws(why) if why is not None else "", condition=c, procedure=p,
                settings={str(k).strip(): _num(v) for k, v in s.items()}, ops=canon_ops(o))
    extra = {k: v for k, v in obj.items() if k not in ("why", "condition", "procedure", "settings", "ops")}
    if extra:
        cand["extra"] = extra
    return cand, None


DENY = [
    (r"\b(option|answer|letter|choice)\s*\(?(?-i:[A-H])\)?\s*(is|=|:)", "option letter given as an answer"),
    (r"\b(answer|option|choice|letter)\s*(is|=|:)\s*\(?(?-i:[A-H])\)?(?![A-Za-z])", "option letter given as an answer"),
    (r"\b(answer|choose|pick|select)\s+(?:option\s+)?\(?(?-i:[A-H])\)?(?![A-Za-z0-9])", "option letter given as an answer"),
    (r"\b(answer|count|is|equals|total)\s*(is\s*)?[:=]?\s*\d+\b", "numeral next to answer/count/is"),
    (r"\b(?:standard|always|usually|typically|normally|by default)\b[^.\n]{0,60}\d+", "fixed value for a kind of object or scene"),
    (r"\b[A-Za-z][A-Za-z ]{1,20}\s*=\s*\d+\b(?=[^\n]*[a-z])", "fixed value for a kind of object or scene"),
    (r"\b\d+\s+(?:rings|legs|columns|squares|pieces|sides|wheels|players)\b", "fixed value for a kind of object or scene"),
]
DENY_RE = [(re.compile(p, re.I), why) for p, why in DENY]

EFFORT = [
    r"\beffort\b",
    r"\b(think|reason|deliberate|ponder)\s+(?:much\s+|a\s+lot\s+|a\s+bit\s+|a\s+little\s+|even\s+|far\s+)?"
    r"(longer|shorter|harder|less|more|briefly|at\s+length|extensively|exhaustively|for\s+longer|deeper|deeply)\b",
    r"\b(more|less|longer|shorter|extended|extensive|extra|additional|minimal|maximum|maximal|brief|short|deeper)\s+"
    r"(thinking|reasoning|deliberation|thought)\b",
    r"\b(spend|take|use|allow|allocate)\s+(?:\w+\s+){0,2}?(more|less|extra|additional|minimal|little|longer|the\s+full)\s+"
    r"(time|tokens?|thinking|reasoning|thought)\b",
    r"\b(over|under)-?think\w*",
    r"\b(token|thinking|reasoning)\s+budget\b",
    r"\b(thinking|reasoning|thoughts?)\s+(short|brief|concise|minimal)\b",
]
EFFORT_RE = [re.compile(p, re.I) for p in EFFORT]
ID_TOKEN = re.compile(r"[A-Za-z0-9_.:\-]+(?:/[A-Za-z0-9_.:\-]+)+")
PIC_TOKEN = re.compile(r"(?<![0-9a-f])[0-9a-f]{24}(?![0-9a-f])")

_META = {}


def _meta(rows):
    key = (id(rows), len(rows))
    m = _META.get(key)
    if m is not None and m["rows"] is rows:
        return m
    R = list(rows.values())
    benches = sorted({str(r["bench"]) for r in R}, key=len, reverse=True)
    phr = {}
    for r in R:
        c = str(r.get("category") or "")
        p = _cat_norm(c)
        if len(p.split(" ")) >= CATEGORY_MIN_WORDS:
            phr.setdefault(p, c)
    cats = sorted(phr.items(), key=lambda x: len(x[0]), reverse=True)
    show = [r for r in R if r.get(ROLE_FIELD) in PROPOSER_ROLES]
    other = [r for r in R if r.get(ROLE_FIELD) not in PROPOSER_ROLES]
    show_txt = {_ws(r["prompt"]) for r in show}
    show_grp = {_ws(r["group"]) for r in show if r.get("group")}
    leak = {}
    for r in other:
        t = _ws(r["prompt"])
        if len(t) >= MIN_TEXT and t not in show_txt:
            leak.setdefault(t, ("question text", r["id"]))
        g = _ws(r.get("group") or "")
        if len(g) >= MIN_TEXT and g not in show_grp:
            leak.setdefault(g, ("group path", r["id"]))
    show_imgs = {pth for r in show for pth in (r.get("image_paths") or [])}
    for r in other:
        for pth in r.get("image_paths") or []:
            if len(pth) >= MIN_TEXT and pth not in show_imgs:
                leak.setdefault(_ws(pth), ("image path", r["id"]))
    allowed = {t for t in show_txt | show_grp if len(t) >= MIN_TEXT}

    def index(texts):
        ix = collections.defaultdict(list)
        for t in texts:
            ix[t[:ANCHOR]].append(t)
        return dict(ix)

    m = dict(
        rows=rows, show_imgs=show_imgs,
        bench_re=re.compile(r"\b(" + "|".join(map(re.escape, benches)) + r")\b", re.I) if benches else None,
        cats=cats,
        ids={r["id"] for r in R}, other_ids={r["id"] for r in other},
        other_pics={r["pic"] for r in other if r.get("pic")} - {r["pic"] for r in show if r.get("pic")},
        leak=leak, leak_ix=index(leak), allow_ix=index(allowed),
    )
    _META.clear()
    _META[key] = m
    return m


def _cat_norm(s):
    return CAT_SEP.sub(" ", str(s).lower()).strip()


def _ids_in(text, idset):
    out = set()
    for tok in ID_TOKEN.findall(text):
        parts = tok.split("/")
        for a, b in zip(parts, parts[1:]):
            for x in {f"{a}/{b}", f"{a.lstrip('.:-')}/{b.rstrip('.:-')}", f"{a}/{b.rstrip('.:-')}"}:
                if x in idset:
                    out.add(x)
    return out


WORD = re.compile(r"[a-z0-9]+")


def card_words(rows, card_ids):
    ids = list(card_ids or [])
    if not ids:
        return set()
    cnt = collections.Counter()
    for i in ids:
        cnt.update(set(WORD.findall(str(rows[i].get("prompt", "")).lower())))
    return {w for w, n in cnt.items() if 2 * n >= len(ids)}


def _label_form(t, c):
    ws = [w for w in CAT_SEP.split(str(c).strip()) if w]
    if not ws:
        return False
    b0, b1 = r"(?<![\w])", r"(?![\w])"
    if re.search(b0 + re.escape(str(c).strip()) + b1, t):
        return True
    if len(ws) > 1 and any(("_" in x.group(0) or "/" in x.group(0)) for x in
                           re.finditer(b0 + r"(?:\s*[_/]\s*|\s+)".join(map(re.escape, ws)) + b1, t, re.I)):
        return True
    if any(ch.isupper() for ch in str(c)) and re.search(b0 + r"\s+".join(map(re.escape, ws)) + b1, t):
        return True
    return False


def leak_precheck(text, rows, allowed_words=None):
    t = str(text or "")
    m = _meta(rows)
    aw = allowed_words or set()
    hits = [why for rx, why in DENY_RE if rx.search(t)]
    if m["bench_re"] is not None and any(x.group(0).lower() not in aw for x in m["bench_re"].finditer(t)):
        hits.append("bench name")
    low = _cat_norm(t)
    for p, c in m["cats"]:
        if p in low and _label_form(t, c):
            if aw and all(w in aw for w in WORD.findall(p)):
                continue
            hits.append(f"category name {c!r}")
    if _ids_in(t, m["ids"]):
        hits.append("row id")
    return list(dict.fromkeys(hits))


def effort_hits(text):
    return [f"asks for more or less thinking ({rx.search(text).group(0)!r}); effort is fixed" for rx in EFFORT_RE if rx.search(str(text or ""))]


def ops_problems(ops):
    if ops is None:
        return []
    if not isinstance(ops, list):
        return ["ops: must be a list of objects"]
    P, seen = [], collections.Counter()
    for o in ops:
        if not isinstance(o, dict):
            P.append(f"ops: {str(o)[:40]!r} is not an object")
            continue
        o = _norm_op(o)
        name = o.get("op")
        if name not in OPS:
            P.append(f"ops: unknown op {name!r} (the ops are {', '.join(OPS_ORDER)})")
            continue
        seen[name] += 1
        for k in sorted(set(o) - {"op"} - set(OPS[name])):
            P.append(f"ops: {name} has no field {k!r}")
        for k, vals in OPS[name].items():
            if k not in o:
                P.append(f"ops: {name} needs {k}, one of {vals}")
            elif not _in_menu(o[k], vals):
                P.append(f"ops: {name} {k} = {o[k]!r} is not one of {vals}")
    for name, n in seen.items():
        if n > 1:
            P.append("ops: more than one tiles" if name == "tiles" else f"ops: {name} is given {n} times")
    return P


def validate(cand, tip, menu, rows, allowed_words=None):
    if not isinstance(cand, dict):
        return ["no candidate"]
    P = []
    for k in cand.get("extra") or {}:
        P.append("effort is fixed and is not a knob" if "effort" in str(k).lower() else f"unknown field {k!r}")
    c, p, s = cand.get("condition"), cand.get("procedure"), cand.get("settings")
    if not isinstance(c, str) or not 1 <= len(c.strip()) <= COND_MAX:
        P.append(f"condition length {len(c.strip()) if isinstance(c, str) else 'n/a'} not in 1..{COND_MAX}")
    if p is not None and not isinstance(p, str):
        P.append("procedure must be a string or null")
        p = None
    if isinstance(p, str) and not p.strip():
        p = None
    if p is not None and len(p) > PROC_MAX:
        P.append(f"procedure has {len(p)} chars > {PROC_MAX}")
    if s is None:
        s = {}
    if not isinstance(s, dict):
        P.append("settings must be an object")
        s = {}
    good = {}
    for k, v in s.items():
        kl = str(k).lower()
        if "effort" in kl or "thinking" in kl or "reasoning" in kl:
            P.append(f"setting {k!r}: effort is fixed and is not a knob")
        elif k not in menu:
            P.append(f"unknown knob {k!r}")
        elif not any(_same(v, x) for x in menu[k]["values"]):
            P.append(f"{k} = {v!r} is not one of {menu[k]['values']}")
        else:
            good[k] = v
    ops = cand.get("ops")
    op_probs = ops_problems(ops)
    P += op_probs
    if p is None and not s and not ops:
        P.append("empty payload: needs a procedure, settings or ops")
    else:
        base, entries = _tip_view(tip, menu)
        same_set = len(good) == len(s) and all(_same(v, base[k]) for k, v in good.items())
        if p is None and not ops and good and same_set:
            P.append("no change vs the tip: every setting equals the tip's")
        elif p is None and ops and not op_probs and same_set:
            always = merge_ops([e.get("ops") for e in entries if is_always(e.get("condition"))])
            if merge_ops([always, ops]) == always:
                P.append("no change vs the tip: " + ("every setting equals the tip's and " if s else "")
                         + "the tip applies these ops on every question already")
        key = entry_key(dict(condition=c, procedure=p, settings=s, ops=ops))
        P += [f"same as tip entry {j}" for j, e in enumerate(entries, 1) if entry_key(e) == key]
    for what, t in (("condition", c), ("procedure", p)):
        if isinstance(t, str) and t.strip():
            P += [f"{what}: {h}" for h in effort_hits(t)]
            P += [f"{what}: leakage precheck: {h}" for h in leak_precheck(t, rows, allowed_words)]
    return P


def check_answer(record, tip, menu, rows, card=None):
    if record is None:
        return None, "unanswered"
    cand, err = parse(record.get("content") or "")
    if cand is None:
        return None, err
    probs = validate(cand, tip, menu, rows, allowed_words=card_words(rows, card) if card else None)
    if probs:
        return cand, "invalid: " + "; ".join(probs)
    return cand, None


def _scan(req, ix):
    get, K = ix.get, ANCHOR
    for i in range(len(req) - K + 1):
        lst = get(req[i:i + K])
        if lst:
            for t in lst:
                if req.startswith(t, i):
                    yield i, i + len(t), t


def _flatten(content, imgs):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f"message content must be a str or a list of parts, not {type(content).__name__}")
    texts = []
    for x in content:
        t = x.get("type") if isinstance(x, dict) else None
        if t == "text":
            texts.append(str(x.get("text", "")))
        elif t == "image":
            imgs.append(x.get("path"))
        else:
            raise ValueError(f"unknown content part {str(x)[:80]}")
    return "".join(texts)


def guard(system, user, rows):
    m = _meta(rows)
    imgs = []
    raw = f"{_flatten(system, imgs)}\n{_flatten(user, imgs)}"
    badimg = [x for x in imgs if x not in m["show_imgs"]]
    if badimg:
        raise LeakError(f"image parts that are not images of SHOW rows: {len(badimg)} (e.g. {str(badimg[0])[-60:]})")
    ids = _ids_in(raw, m["other_ids"])
    if ids:
        raise LeakError(f"non-show row ids in request: {sorted(ids)[:5]} ({len(ids)})")
    pics = set(PIC_TOKEN.findall(raw)) & m["other_pics"]
    if pics:
        raise LeakError(f"non-show picture keys in request: {sorted(pics)[:3]} ({len(pics)})")
    req = _ws(raw)
    hits = list(_scan(req, m["leak_ix"]))
    if hits:
        spans = [(a, b) for a, b, _ in _scan(req, m["allow_ix"])]
        for a, b, t in hits:
            if not any(x <= a and b <= y for x, y in spans):
                what, rid = m["leak"][t]
                raise LeakError(f"{what} of non-show row {rid!r} in request (chars {a}-{b})")
    return dict(chars=len(raw), checked_texts=len(m["leak"]), excused=len(hits), images=len(imgs))
