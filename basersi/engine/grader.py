import math
import re
import zlib

from basersi import config

_LOOP = config.load("engine")["grader"]
ANSWER_MATCH = config.load("benchmarks").get("answer_match") or {}

WORDNUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                      "fifteen sixteen seventeen eighteen nineteen twenty".split())}
FILLER = {"row", "rows", "column", "columns", "col", "and", "the", "th", "st", "nd", "rd", "is", "are", "answer", "final"}


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        i = s.rfind("\\fbox")
        if i < 0:
            return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
        k += 1
    return None


def clean(a):
    a = re.sub(r"\\(?:text|textbf|mathrm|mathbf|operatorname)\{([^{}]*)\}", r"\1", a)
    a = a.replace("$", "").replace("**", "").replace("\\%", "%").replace("\\,", "").replace("\\!", "").replace("\\ ", " ")
    a = a.replace("\u2212", "-").replace("\u2013", "-").replace("\u2192", "-").replace("\\to", "-").replace("\\rightarrow", "-")
    return a.strip().strip(".").strip()


def final_answer(content):
    b = last_boxed(content)
    if b is not None:
        return clean(b)
    m = re.findall(r"(?:final answer|answer)\s*(?:is|:)\s*\**\s*([^\n]+)", content, re.I)
    return clean(m[-1]) if m else None


def norm_txt(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def mc_letter(a, options, content):
    if a is not None:
        for i, o in enumerate(options):
            if norm_txt(a) == norm_txt(str(o)):
                return "ABCDEFGHIJ"[i]
        m = re.match(r"^(?:option\s*)?\(?([A-Ja-j])\)?(?=$|[\s.:)\],])", a.strip(), re.I)
        if m and (m.group(1).isupper() or len(a.strip()) <= 3):
            return m.group(1).upper()
        return None
    m = re.findall(r"answer is:?\s*\**\s*\(?([A-J])\)?\b", content)
    return m[-1] if m else None


def nums(a):
    if a is None:
        return []
    s = a.lower().replace("\u2212", "-").replace("%", "").replace("\\%", "").replace("$", "").strip()
    if "=" in s:
        s = s.split("=")[-1]
    s = re.sub(r"\\[dt]?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", lambda m: _div(m.group(1), m.group(2)), s)
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*", s)
    if m:
        s = _div(m.group(1), m.group(2))
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = re.sub(r"([a-z])\s*\^\s*\{?-?\d+\}?", r"\1", s)
    s = re.sub(r"(-?\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|x|\*|×)\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?", lambda m: _pow10(m.group(1), m.group(2)), s)
    s = re.sub(r"(?<![\d.])10\s*\^\s*\{?\s*(-?\d+)\s*\}?", lambda m: _pow10("1", m.group(1)), s)
    if "nan_big" in s:
        return []
    out = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", s)]
    if not all(math.isfinite(x) for x in out):
        return []
    if out:
        return out
    w = re.findall(r"[a-z]+", s)
    return [float(WORDNUM[w[0]])] if w and w[0] in WORDNUM else []


def _pow10(a, e):
    try:
        e = int(e)
        if e > 400:
            return "nan_big"
        v = float(a) * 10.0 ** e
        return repr(v) if math.isfinite(v) else "nan_big"
    except (ValueError, OverflowError):
        return "nan_big"


def _div(p, q):
    try:
        return repr(float(p) / float(q))
    except (ValueError, ZeroDivisionError):
        return "nan_div"


def to_num(a):
    v = nums(a)
    return v[0] if v and len(set(v)) == 1 else None


def num_ok(pred, gold, precision=None):
    if pred is None:
        return False
    g = float(gold) if not isinstance(gold, str) or re.fullmatch(r"\s*-?\d+(\.\d+)?(e[-+]?\d+)?\s*", gold, re.I) else to_num(gold)
    if g is None or not math.isfinite(g) or not math.isfinite(pred):
        return False
    if precision is not None:
        return round(pred, int(precision)) == round(g, int(precision))
    return abs(pred - g) <= 1e-6 * max(1.0, abs(g))


def canon(s):
    s = clean(s).lower()
    if "=" in s and s.split("=")[-1].strip():
        s = s.split("=")[-1]
    pairs = re.findall(r"([a-z0-9]+)\s*-\s*([a-z0-9]+)", s)
    rest = re.sub(r"([a-z0-9]+)\s*-\s*([a-z0-9]+)", " ", s)
    if pairs and not re.findall(r"[a-z0-9]", re.sub(r"\b(?:%s)\b" % "|".join(FILLER), " ", rest)):
        return ("pairs", frozenset(pairs))
    toks = [t for t in re.findall(r"[a-z]+|\d+(?:\.\d+)?", s) if t not in FILLER]
    toks = [str(WORDNUM[t]) if t in WORDNUM else t for t in toks]
    return ("toks", tuple(toks)) if toks else ("raw", s.replace(" ", ""))


GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ϵ": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
         "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau",
         "υ": "upsilon", "φ": "phi", "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega", "Γ": "gamma", "Δ": "delta", "Θ": "theta",
         "Λ": "lambda", "Ξ": "xi", "Π": "pi", "Σ": "sigma", "Φ": "phi", "Ψ": "psi", "Ω": "omega"}


def term_norm(s):
    s = clean(s)
    for k, v in GREEK.items():
        s = s.replace(k, v)
    s = s.lower().replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-").replace("\u2014", "-")
    s = re.sub(r"\\(?:left|right|mathcal|mathit|mathsf|mathbb|bm|boldsymbol|hat|bar|tilde|vec)\b", "", s)
    s = s.replace("\\", "")
    s = re.sub(r"^\s*the\s+", "", s)
    return re.sub(r"[^a-z0-9+\-=<>.%/*]", "", s).strip(".")


def bench_of(r):
    return r.get("bench") or str(r.get("id") or "").split("/")[0]


def uses_term_match(r):
    return r.get("category") in (ANSWER_MATCH.get(bench_of(r)) or {}).get("term_categories", ())


def term_eq(a, g):
    if term_norm(a) == term_norm(g) and term_norm(g):
        return True
    sp = lambda x: {t for t in (term_norm(y) for y in re.split(r",|;|&|\band\b", clean(x), flags=re.I)) if t}
    ta, tg = sp(a), sp(g)
    return len(tg) > 1 and ta == tg


def correct(r):
    if r["finish"] != "stop" or not r["content"]:
        return False, None
    a = final_answer(r["content"])
    k = r["kind"]
    if k == "mc":
        p = mc_letter(a, r.get("options") or [], r["content"])
        return p == r["answer"], p
    if k == "num":
        g = nums(str(r["answer"]))
        if len(g) > 1:
            v = nums(a)
            return len(v) == len(g) and all(num_ok(x, y) for x, y in zip(v, g)), (",".join(f"{x:g}" for x in v) if v else None)
        p = to_num(a)
        return num_ok(p, r["answer"], r.get("precision")), p
    if k == "yesno":
        p = None
        if a is not None:
            m = re.match(r"^\W*(yes|no)\b", a.lower())
            p = m.group(1) if m else None
        return p == r["answer"], p
    if a is None or not a.strip():
        return False, None
    if uses_term_match(r):
        return term_eq(a, str(r["answer"])), a
    cg, cp = canon(r["answer"]), canon(a)
    if cg == cp:
        return True, a
    if cg[0] == cp[0] == "pairs":
        left, right = {x for x, _ in cg[1]}, {y for _, y in cg[1]}
        if not left & right:
            return {frozenset(t) for t in cg[1]} == {frozenset(t) for t in cp[1]}, a
    return False, a


def looped(r):
    tail = ((r.get("reasoning") or "") + (r.get("content") or ""))[-int(_LOOP["loop_tail_chars"]):]
    return len(tail) > int(_LOOP["loop_min_chars"]) and len(zlib.compress(tail.encode())) / len(tail.encode()) < float(_LOOP["loop_max_compression"])


NO_ANSWER_CLASS = "none"


def grade_read(row, content, reasoning, finish, tokens, sample=0):
    r = {"id": row["id"], "bench": row.get("bench"), "kind": row["kind"], "answer": row["answer"], "sample": sample, "reasoning": reasoning or "",
         "content": content or "", "finish": finish, "tokens": tokens, "options": row.get("options") or [],
         "category": row.get("category"), "images": list(row.get("image_paths") or [])}
    if row.get("precision") is not None:
        r["precision"] = row["precision"]
    err, pred = None, None
    if finish != "stop" or not content:
        ok = False
    else:
        try:
            ok, pred = correct(r)
            ok = bool(ok)
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {str(e)[:160]}"
    try:
        loop = bool(looped(r))
    except Exception as e:
        loop, err = False, err or f"{type(e).__name__}: {str(e)[:160]}"
    if pred is not None and not isinstance(pred, (str, int, float)):
        pred = str(pred)
    return ok, loop, err, pred


def _record_for_grader(rec, row):
    r = {"id": row["id"], "bench": row.get("bench"), "kind": row["kind"], "answer": row["answer"], "options": row.get("options") or [],
         "category": row.get("category"), "content": rec.get("content") or "", "finish": rec.get("finish")}
    if row.get("precision") is not None:
        r["precision"] = row["precision"]
    return r


def _graded(rec, row):
    if rec.get("finish") != "stop" or not rec.get("content"):
        return False, None
    try:
        ok, p = correct(_record_for_grader(rec, row))
    except Exception:
        return False, None
    if p is not None and not isinstance(p, (str, int, float)):
        p = str(p)
    return bool(ok), p


def graded_record(rec, row):
    ok, p = _graded(rec, row)
    if "pred" in rec:
        p = rec["pred"]
    if rec.get("ok") is not None:
        ok = bool(rec["ok"])
    return (False if p is None else ok), p


def answer_class(p, row):
    if p is None:
        return NO_ANSWER_CLASS
    k = row.get("kind")
    if k in ("mc", "yesno"):
        return ("v", str(p))
    if k == "num":
        if isinstance(p, (int, float)) and not isinstance(p, bool):
            x = float(p)
            return ("num", round(x, int(row["precision"])) if row.get("precision") is not None else x)
        return ("v", str(p))
    s = str(p)
    if uses_term_match(row):
        return ("term", term_norm(s))
    try:
        return ("canon", canon(s))
    except Exception:
        return ("v", s)


def record_pred(rec, row):
    if "pred" in rec:
        return rec["pred"]
    return _graded(rec, row)[1]
