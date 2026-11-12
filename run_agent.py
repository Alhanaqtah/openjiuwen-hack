"""Budget-aware harness for openJiuwen Challenge 2026.

python run_agent.py --tasks-dir /data/tasks
env: LLM_BASE_URL, LLM_API_KEY, BUDGET (output-token equivalents; input counts 1/10)
"""
import argparse
import ast
import asyncio
import collections
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

_LAUNCH_DIR = os.getcwd()
# openJiuwen writes ./logs on import; the container root is read-only, so work from a tmp dir
_WORKDIR = Path(os.environ.get("HARNESS_WORKDIR", Path(tempfile.gettempdir()) / "harness"))
_WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(_WORKDIR)

def _quiet_openjiuwen_logs():
    """openJiuwen echoes every LLM request/response to stdout; keep its logs in files only, warnings and up."""
    try:
        from openjiuwen.core.common.logging.log_config import log_config
        from openjiuwen.core.common.logging.manager import LogManager
        cfg = log_config.get_snapshot()
        cfg["level"] = "WARNING"
        cfg["propagate"] = False
        for k in list(cfg):
            if k == "output" or k.endswith("_output"):
                cfg[k] = ["file"]
        log_config.load_from_dict(cfg)
        LogManager.reset()
    except Exception as e:  # logging must never break the run
        print("openjiuwen log config not adjusted:", e)


_quiet_openjiuwen_logs()

from openjiuwen.core.foundation.llm.model import Model  # noqa: E402
from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig, ModelRequestConfig  # noqa: E402

log = logging.getLogger("harness")

FAST = os.environ.get("MODEL_FAST", "qwen3.8-27b-noreason")      # primary model
STRONG = os.environ.get("MODEL_STRONG", "qwen3.6-35b-a3b-noreason")  # second opinion / retry
REASON = os.environ.get("MODEL_REASON", "qwen3.8-27b")  # tie-breaker only (reasoning tokens are expensive)
VERIFY = os.environ.get("VERIFY", "1") == "1"  # phase 2: second opinion on answered tasks while budget remains
VERIFY_KEEP = float(os.environ.get("VERIFY_KEEP", "0.5"))  # share of budget never spent on verification
VERIFY_TOP = float(os.environ.get("VERIFY_TOP", "0.5"))  # verify only this top share of answered tasks by reward
PRECHECK = os.environ.get("PRECHECK", "0") == "1"  # gate every compute task (hurts recall; off)
ZERO_GUARD = os.environ.get("ZERO_GUARD", "1") == "1"  # re-check tasks whose computed answer is 0
PARALLEL = int(os.environ.get("PARALLEL", "4"))
LLM_SLOTS: asyncio.Semaphore  # concurrent HTTP calls, created in run()
WALL_LIMIT = float(os.environ.get("WALL_LIMIT", str(55 * 60)))
UNSOLVABLE = "UNSOLVABLE"
UNSOLVABLE_RE = re.compile(r"UNSOL|IMPOSSIB|CANNOT|NOT POSSIBLE|UNKNOWN|N/A|NOT (?:ENOUGH|AVAILABLE|FOUND|PRESENT|GIVEN|SPECIFIED)|INSUFFICIENT|NO (?:SUCH|INFORMATION|DATA)", re.I)


def is_unsolvable(text: str) -> bool:
    return bool(UNSOLVABLE_RE.search(text or ""))
CHARS_PER_TOKEN = 3.2


def cost(inp: int, out: int) -> float:
    return inp / 10 + out


# ---------------------------------------------------------------- budget

class Ledger:
    """Tracks spend; reserves the worst case of a call before making it."""

    def __init__(self, budget: float, margin: float):
        self.budget = budget
        self.margin = margin
        self.spent = 0.0
        self.reserved = 0.0

    @property
    def free(self) -> float:
        return self.budget - self.margin - self.spent - self.reserved

    def reserve(self, amount: float) -> bool:
        if amount > self.free:
            return False
        self.reserved += amount
        return True

    def settle(self, reserved: float, actual: float):
        self.reserved -= reserved
        if self.reserved < 1e-6:  # float drift must not look like money still in flight
            self.reserved = 0.0
        self.spent += actual

    @property
    def in_flight(self) -> bool:
        return self.reserved > 1e-6


class BudgetExhausted(Exception):
    pass


class LLM:
    def __init__(self, ledger: Ledger, trace: list):
        self.ledger = ledger
        self.trace = trace
        self.client_cfg = ModelClientConfig(
            client_provider="OpenAI",
            api_key=os.environ["LLM_API_KEY"],
            api_base=os.environ["LLM_BASE_URL"],
            timeout=180,
            max_retries=2,
            verify_ssl=os.environ.get("LLM_VERIFY_SSL", "1") != "0",
        )
        self.models: dict[str, Model] = {}

    def _model(self, name: str) -> Model:
        if name not in self.models:
            self.models[name] = Model(self.client_cfg, ModelRequestConfig(model=name, temperature=0))
        return self.models[name]

    calls = 0
    FAIL_AFTER = int(os.environ.get("HARNESS_FAIL_AFTER", "0"))  # dev: simulate the proxy dying

    async def _invoke(self, messages, model, max_tokens):
        LLM.calls += 1
        if LLM.FAIL_AFTER and LLM.calls > LLM.FAIL_AFTER:
            raise RuntimeError("simulated proxy failure: budget exhausted (402)")
        delay = 2.0
        for attempt in range(8):
            try:
                async with LLM_SLOTS:
                    return await self._model(model).invoke(messages, model=model, max_tokens=max_tokens,
                                                            temperature=0)
            except Exception as e:
                msg = str(e)
                transient = any(k in msg for k in ("429", "RateLimit", "Too Many", "502", "503", "504",
                                                   "Timeout", "timed out", "Connection"))
                if not transient or attempt == 7:
                    raise
                log.info("retry %d after %s", attempt + 1, msg[:120])
                await asyncio.sleep(delay)
                delay = min(delay * 1.7, 20)

    async def ask(self, messages: list[dict], *, model: str, max_tokens: int, tag: str, want_truncated: bool = False):
        prompt_tokens = sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN + 20 * len(messages)
        worst = cost(prompt_tokens * 1.3, max_tokens)
        while not self.ledger.reserve(worst):
            if not self.ledger.in_flight:  # nothing in flight will free money: really out of budget
                raise BudgetExhausted(f"{tag}: need {worst:.0f}, free {self.ledger.free:.0f}")
            await asyncio.sleep(0.3)
        t0 = time.time()
        actual = cost(prompt_tokens, 0)  # failed call: assume the prompt was billed
        try:
            resp = await self._invoke(messages, model, max_tokens)
            u = resp.usage_metadata
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            if u and (u.input_tokens or u.output_tokens):
                actual = cost(u.input_tokens, u.output_tokens)
            else:  # no usage reported: estimate conservatively
                actual = cost(prompt_tokens * 1.3, len(text) / 2.5 + len(resp.reasoning_content or "") / 2.5)
            truncated = bool(u and u.output_tokens >= max_tokens) or \
                getattr(resp, "finish_reason", None) == "length"
            self.trace.append({"tag": tag, "model": model, "in": u.input_tokens if u else None,
                               "out": u.output_tokens if u else None, "cost": round(actual, 1),
                               "sec": round(time.time() - t0, 1), "trunc": truncated})
            return (text, truncated) if want_truncated else text
        finally:
            self.ledger.settle(worst, actual)


# ---------------------------------------------------------------- tasks

@dataclass
class Task:
    dir: Path
    data: dict
    est: float = 0.0
    status: str = "pending"
    answer: str | None = None
    trace: list = field(default_factory=list)

    @property
    def id(self): return self.data.get("task_id", self.dir.name)
    @property
    def type(self): return self.data.get("task_type", "")
    @property
    def reward(self): return float(self.data.get("reward", 1) or 1)
    @property
    def ws(self):
        ws = self.dir / "workspace"
        return ws if ws.is_dir() else self.dir
    @property
    def instruction(self): return self.data.get("instruction", "")


def strip_boilerplate(instr: str) -> str:
    """The 'write answer.txt' tail is handled by the harness; dropping it saves tokens and confusion."""
    instr = re.sub(r"\s*When you are done, write ONLY.*$", "", instr, flags=re.S)
    instr = re.sub(r"\s*Work only inside the current directory\.", "", instr)
    return instr.strip()


def ws_files(ws: Path) -> list[Path]:
    if not ws.is_dir():
        return []
    return sorted(p for p in ws.rglob("*") if p.is_file() and not p.name.startswith(".")
                  and p.name not in ("task.json", "answer.txt"))


def read(p: Path, limit: int = 200_000) -> str:
    return p.read_text(errors="replace")[:limit]


def write_answer(task: Task, answer: str):
    answer = answer.strip()
    for target in (task.dir / "answer.txt", task.ws / "answer.txt"):
        try:
            target.write_text(answer + "\n")
            log.info("%s -> %s: %r", task.id, target, answer[:80])
            return
        except OSError as e:
            log.warning("%s: cannot write %s: %s", task.id, target, e)


KB_PRELUDE = r'''
import os, re, sys, json, csv, collections, itertools, math
_ROOT = globals().get("_ROOT", ".")
kb = {}
for _r, _d, _fs in os.walk(_ROOT):
    _d[:] = [x for x in _d if not x.startswith(".") and x != "__pycache__"]
    for _f in sorted(_fs):
        _p = os.path.relpath(os.path.join(_r, _f), _ROOT)
        if _f.startswith(".") or _f in ("task.json", "answer.txt") or os.path.getsize(os.path.join(_ROOT, _p)) > 2_000_000:
            continue
        try:
            kb[_p] = open(os.path.join(_ROOT, _p), encoding="utf-8", errors="replace").read()
        except OSError:
            pass
class _Missing(str):
    """stands for an absent record/field: any lookup returns itself, it is falsy, behaves as an empty string
    in regexes and prints as UNSOLVABLE"""
    def __new__(cls): return str.__new__(cls, "")
    def __getitem__(self, k): return self
    def get(self, k, d=None): return self
    def __getattr__(self, k): return self
    def __call__(self, *a, **k): return self
    def __bool__(self): return False
    def __iter__(self): return iter(())
    def __len__(self): return 0
    def __eq__(self, o): return isinstance(o, _Missing)
    def __hash__(self): return 0
    def __str__(self): return "UNSOLVABLE"
    def __format__(self, spec): return "UNSOLVABLE"
    __repr__ = __str__
    def lower(self): return self
    def strip(self, *a): return self
MISSING = _Missing()
_SMALL = r"(?:\s+(?:by|in|at|on|of|is|was|are|were|the|a|an|to|for|from|as|:|-|=))*"
class Rec(dict):
    """record whose missing fields fall back to a keyword search in its own text: 'born' finds 'born in 1990',
    'headquarters' finds 'headquartered in Riga', 'mentor' finds 'mentored by Ann Lee'"""
    def __missing__(self, k):
        text = dict.get(self, "_text", "")
        stem = re.escape(str(k).strip().lower()[:5])
        if not text or not stem or str(k).startswith("_"): return MISSING
        m = re.search(r"\b" + stem + r"\w*" + _SMALL + r"\s+(\d[\d.,/-]*|[A-Z][\w'&.-]*(?:\s+(?:[A-Z][\w'&.-]*|of|and|the|&))*)", text)
        return m.group(1).rstrip(".,;") if m else MISSING
    def get(self, k, d=None): return self[k] if k in self else (MISSING if d is None else d)
records = {}
_FIELD = re.compile(r"^\s*(?:[-*+]\s*)?\**([A-Za-z][\w /().'-]{0,40}?)\**\s*:\s*\**(.+?)\**\s*$", re.M)
def _put(title, fields, path, text):
    t = str(title).strip().strip("#").strip()
    if not t: return
    key = t if t not in records else f"{t} ({path})"
    records[key] = Rec({**{str(k).strip().lower(): v for k, v in fields.items()}, "_file": path, "_text": text.strip()})
def _walk_json(obj, path):
    if isinstance(obj, dict):
        for nk in ("name", "title", "full_name", "id", "person", "company", "label"):
            if nk in obj and isinstance(obj[nk], (str, int)):
                _put(obj[nk], {k: v for k, v in obj.items() if not isinstance(v, (dict, list))}, path, json.dumps(obj, ensure_ascii=False))
                break
        for v in obj.values():
            _walk_json(v, path)
    elif isinstance(obj, list):
        for v in obj:
            _walk_json(v, path)
for _p, _t in kb.items():
    if _p.endswith((".json", ".jsonl")):
        try:
            _walk_json(json.loads(_t) if _p.endswith(".json") else [json.loads(l) for l in _t.splitlines() if l.strip()], _p)
            continue
        except ValueError:
            pass
    _secs = re.split(r"(?m)^(?=#{1,6}\s)", _t)
    _any = False
    for _sec in _secs:
        _m = re.match(r"#{1,6}\s*(.+)", _sec)
        _fields = {k: v for k, v in _FIELD.findall(_sec)}
        if _m or _fields:
            _any = True
            _put(_m.group(1) if _m else os.path.splitext(os.path.basename(_p))[0].replace("_", " ").replace("-", " "), _fields, _p, _sec)
    if not _any and _t.strip():  # free text: one record per file, titled by the leading proper name or the file name
        _lead = re.match(r"\s*([A-Z][\w'-]+(?:\s+[A-Z][\w'&.-]+){0,3})", _t)
        _put(_lead.group(1) if _lead else os.path.splitext(os.path.basename(_p))[0].replace("_", " ").replace("-", " "), {}, _p, _t)
def _title(n):
    for k, v in records.items():
        if k.lower() == n or k.lower().startswith(n + " ("): return v
def _starts(n):
    hits = [v for v in records.values() if v["_text"].lower().lstrip("#*- ").startswith(n)]
    return hits[0] if len(hits) == 1 else None
def rec(name):
    """record by title (case-insensitive exact), else the unique record whose text starts with / mentions the name;
    a too-long name (greedy capture) is shortened word by word until a title matches"""
    if name is None or isinstance(name, _Missing): return MISSING
    n = re.sub(r"\s+", " ", str(name).strip().strip(".,;:")).lower()
    if not n: return MISSING
    words = n.split()
    prefixes = [" ".join(words[:k]) for k in range(len(words), 1, -1)] or [n]
    for f in (_title, _starts):
        for pref in prefixes:
            r = f(pref)
            if r: return r
    hits = [v for v in records.values() if n in v["_text"].lower()]
    return hits[0] if len(hits) == 1 else MISSING
def facts(name):
    """sentences from any file that mention the name (for free-text notes)"""
    if isinstance(name, _Missing): return []
    n = str(name).strip().lower()
    return [s.strip() for t in kb.values() for s in re.split(r"(?<=[.!?])\s+|\n", t) if n in s.lower()]
def grep(pattern):
    """[(path, line)] for lines matching the regex (case-insensitive) in any file"""
    rx = re.compile(pattern, re.I)
    return [(p, ln) for p, t in kb.items() for ln in t.splitlines() if rx.search(ln)]
'''


_KB_CACHE: dict[str, dict] = {}


def kb_env(ws: Path) -> dict:
    """The prelude executed in-process for this workspace (records, rec, ...), cached per task."""
    key = str(ws)
    if key not in _KB_CACHE:
        g: dict = {"_ROOT": key}
        try:
            exec(KB_PRELUDE, g)
        except Exception as e:
            g = {"records": {}, "error": str(e)}
        _KB_CACHE[key] = g
    return _KB_CACHE[key]


PLACE_Q = re.compile(r"\b(where|which city|what city|in which city|in what city|which town|which country|located)\b", re.I)
PLACE_FIELDS = ("headquarters", "hq", "city", "location", "located", "based", "town", "country", "office", "address")


def refine_answer(question: str, ans: str, ws: Path) -> str:
    """One hop short is a common model slip: 'where is X's employer' answered with the employer's name.
    If the answer is itself a record title and the question asks for a place, take the record's place field."""
    if not PLACE_Q.search(question):
        return ans
    g = kb_env(ws)
    r = g["records"].get(ans) or next((v for k, v in g["records"].items() if k.lower() == ans.lower()), None)
    if not r:
        return ans
    for f in PLACE_FIELDS:
        v = r[f]  # Rec: exact field, else keyword fallback in the record text, else MISSING (empty)
        if isinstance(v, str) and v:
            log.info("refine: %r -> %r via field %s", ans, v, f)
            return v
    return ans


def kb_summary(ws: Path) -> str:
    """Describe what the parser produced for this knowledge base (record count, field names)."""
    g = kb_env(ws)
    if "error" in g:
        return f"(kb parse failed: {g['error']})"
    recs = g["records"]
    fields = collections.Counter(k for r in recs.values() for k in r if not k.startswith("_"))
    if not recs:
        return "records is EMPTY: no files found."
    if not fields:
        return (f"records: {len(recs)} entries titled by file name, with NO key:value fields (free-text notes). "
                "Facts are inside sentences: use grep(regex) / rec(name)['_text'] with regular expressions.")
    sample = ", ".join(f"{k}({n})" for k, n in fields.most_common(14))
    titles = list(recs)[:3]
    return (f"records: {len(recs)} entries, e.g. titles {titles}; fields seen: {sample}. "
            "Use exactly these field names; a fact that is not a field may still be in rec(name)['_text'].")


def same_answer(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return a is b
    return normalize_number(a).lower().strip() == normalize_number(b).lower().strip()


def run_python(code: str, cwd: Path, timeout: int = 30, prelude: str = "") -> tuple[int, str, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=os.environ.get("TMPDIR", "/tmp")) as f:
        f.write(prelude + code)
        path = f.name
    try:
        p = subprocess.run([sys.executable, "-I", path], cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr[-2000:]
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    finally:
        os.unlink(path)


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, re.S)  # unterminated (truncated) block
    if m:
        return m.group(1)
    return text.strip()


def check_line_says_unsolvable(reply: str) -> bool:
    m = re.search(r"CHECK:\s*(.+)", reply)
    if not m:
        return False
    line = m.group(1)
    if re.match(r"\s*OK\b", line, re.I):
        return False
    return bool(re.search(r"\b(missing|not given|not specified|not provided|not present|not available|unspecified|absent|unknown|no such|does not exist|cannot|unsolvable)\b", line, re.I)) \
        and not re.search(r"\b(all (?:are |quantities are )?(?:given|present|provided|specified)|nothing (?:is )?missing|none missing)\b", line, re.I)


NUMERIC_Q = re.compile(r"\b(how many|how much|how old|how far|how long|what (?:is|was|were) the (?:total|number|count|sum|average|mean|maximum|minimum|highest|lowest|balance|revenue|value|age)|(?:in |during )?(?:what|which) year|birth year|year of birth|year (?:was|did|when|were)|founded|what age|age of|what number|what percentage)\b", re.I)
TEXT_Q = re.compile(r"\b(which|what|in what)\s+(city|town|country|company|organi[sz]ation|employer|region|team|department|name|product|station|colou?r|language)\b|\b(where|whose|whom)\b|^\W*who\b", re.I)


def answer_type_ok(question: str, ans: str) -> bool:
    """Reject answers whose shape contradicts the question (a year for 'which city', a name for 'how many',
    a raw 'Key: value' line): the usual symptom of printing a related fact instead of the one asked for."""
    a = ans.strip()
    is_num = bool(re.fullmatch(r"-?[\d,]+(?:\.\d+)?", a))
    if re.fullmatch(r"[A-Za-z_ ]{2,30}:\s*\S.*", a):
        return False
    if TEXT_Q.search(question) and not NUMERIC_Q.search(question) and is_num:
        return False
    if NUMERIC_Q.search(question) and not TEXT_Q.search(question) and not is_num and not re.search(r"\d", a):
        return False
    return True


def clean_answer(out: str) -> str:
    lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------- estimators

# Output tokens per first call (p75 from our traces) and empirical retry rate per task type
OUT_P75 = {"math": 90, "data_analysis": 90, "search": 110, "coding": 200}
RETRY_P = {"math": 0.05, "data_analysis": 0.08, "search": 0.2, "coding": 0.3}
CHARS_PER_TOKEN_EN = 3.6


def file_preview(ws: Path, budget_chars: int = 3000) -> str:
    parts = []
    for p in ws_files(ws):
        rel = p.relative_to(ws)
        text = read(p, 4000)
        lines = text.splitlines()
        head = "\n".join(lines[:8])
        parts.append(f"--- {rel} ({len(lines)} lines, {p.stat().st_size} bytes)\n{head}")
    s = "\n".join(parts)
    return s[:budget_chars]


def search_context(ws: Path, question: str, limit_chars: int = 24000) -> str:
    files = ws_files(ws)
    total = sum(p.stat().st_size for p in files)
    if total <= limit_chars:
        return "\n\n".join(f"### {p.relative_to(ws)}\n{read(p).strip()}" for p in files)
    # large KB: rank files by overlap with question terms, then follow names mentioned in top hits
    words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", question)}
    texts = {p: read(p, 20000) for p in files}
    def score(p):
        t = (p.name + " " + texts[p]).lower()
        return sum(t.count(w) for w in words) + 5 * sum(w in p.name.lower() for w in words)
    ranked = sorted(files, key=score, reverse=True)
    out, used = [], 0
    for p in ranked:
        chunk = f"### {p.relative_to(ws)}\n{texts[p].strip()}"
        if used + len(chunk) > limit_chars:
            break
        out.append(chunk)
        used += len(chunk)
    return "\n\n".join(out)


def prompt_chars(task: Task) -> int:
    """Size of the first prompt the solver will send (system + user), in characters."""
    instr = strip_boilerplate(task.instruction)
    if task.type == "search":
        return len(SYS_KB) + len(instr) + len(kb_preview(task.ws, instr))
    if task.type == "coding":
        py = min(24000, sum(p.stat().st_size for p in ws_files(task.ws) if p.suffix == ".py"))
        return len(SYS_CODE) + len(instr) + py + 1200  # + failing test output
    return len(SYS_COMPUTE) + len(instr) + len(file_preview(task.ws))


def estimate(task: Task) -> float:
    """Expected spend (budget units) for this task: first call + retry probability. Calibrated on traces."""
    inp = prompt_chars(task) / CHARS_PER_TOKEN_EN + 16
    out = OUT_P75.get(task.type, 150)
    first = cost(inp, out)
    retry = RETRY_P.get(task.type, 0.3)
    return first * (1 + retry * 1.6)  # a retry carries the first exchange again plus the fix


# ---------------------------------------------------------------- solvers

SYS_COMPUTE = (
    "Solve the task by writing a short Python 3 program (stdlib only; csv/json for data files) that runs in the "
    "task directory and prints ONLY the final answer on the last line. Follow the wording literally and in order "
    "(inclusive ranges; 'added'/'received' +, 'charged'/'shipped' -; 'rounded down' = //); integer arithmetic for "
    "whole items; use only columns/fields that exist, never reinterpret one as another quantity (years of "
    f"experience are not age, salary is not bonus, units are not returns). If a needed number, column or value is "
    f"missing or unspecified, print {UNSOLVABLE} instead of guessing or assuming 0.\n"
    "Reply format: first line 'CHECK: OK' or 'CHECK: MISSING <what>', then one ```python block, as few lines as "
    "possible, no comments."
)


SYS_PRECHECK = (
    "You are a strict reviewer. Decide whether the task can be answered EXACTLY from the information given "
    "(the text and the listed data columns). Answer NO if a needed number is unspecified ('some', 'several', "
    "'a few', 'unknown amount', 'a percentage announced'), or the question needs a quantity that no column "
    "or field measures (e.g. age when only years of experience exist, discounts/returns/profit when absent). "
    "Answer YES if everything needed is present. Reply with one word: YES or NO."
)


async def precheck(task: Task, llm: LLM, instr: str, context: str, model: str = FAST) -> bool:
    """Cheap solvability gate (~1 output token). True = looks solvable."""
    msgs = [{"role": "system", "content": SYS_PRECHECK},
            {"role": "user", "content": f"Task: {instr}\n" + (f"\nData available:\n{context}\n" if context else "")}]
    reply = await llm.ask(msgs, model=model, max_tokens=2, tag=f"{task.id}#pre")
    task.trace.append({"precheck": reply})
    return not reply.strip().upper().startswith("NO")


async def solve_compute(task: Task, llm: LLM, models=None) -> str | None:
    models = models or (FAST, STRONG)
    instr = strip_boilerplate(task.instruction)
    preview = file_preview(task.ws)
    if PRECHECK and not await precheck(task, llm, instr, file_preview(task.ws, 600)):
        return None
    user = f"Task:\n{instr}\n" + (f"\nFiles in the current directory (first lines):\n{preview}\n" if preview else "")
    msgs = [{"role": "system", "content": SYS_COMPUTE}, {"role": "user", "content": user}]
    for attempt, model in enumerate(models):
        reply = await llm.ask(msgs, model=model, max_tokens=700 if "noreason" in model else 2500,
                              tag=f"{task.id}#{attempt}")
        if check_line_says_unsolvable(reply):
            task.trace.append({"check": reply[:300]})
            return None
        code = extract_code(reply)
        rc, out, err = run_python(code, task.ws)
        ans = clean_answer(out)
        task.trace.append({"code": code, "rc": rc, "out": out[-500:], "err": err[-500:]})
        if rc == 0 and ans:
            if is_unsolvable(ans):
                return None
            ans = normalize_number(ans)
            if not answer_type_ok(instr, ans):
                task.trace.append({"rejected": ans})
                return None
            # a zero count/sum is the typical symptom of a question about data that is not there
            if ZERO_GUARD and task.type == "data_analysis" and ans in ("0", "0.0") \
                    and not await precheck(task, llm, instr, preview[:600], model=STRONG):
                return None
            return ans
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": f"The program failed (exit {rc}).\nstdout:\n{out[-800:]}\nstderr:\n{err[-1200:]}\nFix it. One ```python block."}]
    return None


def normalize_number(s: str) -> str:
    s = s.strip().strip("`'\"")
    try:
        f = float(s.replace(",", ""))
        if f.is_integer() and re.fullmatch(r"-?[\d,]+(\.0+)?", s):
            return str(int(f))
    except ValueError:
        pass
    return s


SYS_SEARCH = (
    "Answer the question using ONLY the knowledge base below. Facts may require following links between "
    "files (e.g. person -> employer -> company headquarters). Names must match exactly; a similar name is a "
    f"different entity. If the knowledge base does not contain the answer, reply {UNSOLVABLE}. "
    "Reply with the bare answer only (a year, a name, a city...), nothing else."
)


async def solve_search(task: Task, llm: LLM, models=None) -> str | None:
    instr = strip_boilerplate(task.instruction)
    instr = re.sub(r"\s*Use the file tools.*?\.\s*", " ", instr).strip()
    ctx = search_context(task.ws, instr)
    msgs = [{"role": "system", "content": SYS_SEARCH},
            {"role": "user", "content": f"Knowledge base:\n{ctx}\n\nQuestion: {instr}"}]
    reply = await llm.ask(msgs, model=FAST, max_tokens=40, tag=task.id)
    ans = clean_answer(reply).strip(" .`*\"'")
    task.trace.append({"reply": reply})
    if not ans or is_unsolvable(ans) or len(ans) > 80:
        return None
    return ans


SYS_CODE = (
    "Fix bugs in Python source files so the project's tests pass. The grader runs HIDDEN tests broader than the "
    "visible ones (empty inputs, single elements, negatives, duplicates, mixed case, boundaries), so review EVERY "
    "function/method, not only the failing ones, and make each behave as its name and docstring imply. Never touch "
    "the tests. Reply with ```python blocks containing ONLY corrected top-level functions/classes (complete "
    "definitions; a changed method means the whole class); with several source files start each block with "
    "'# file: <path>'. Omit code that is already correct. No prose, no shell commands. If two tests demand "
    f"contradictory results for the same input, reply {UNSOLVABLE}."
)


def is_test_file(p: Path) -> bool:
    return p.suffix == ".py" and (p.name.startswith("test") or p.stem.endswith("_test") or p.stem.endswith("_tests")
                                  or "tests" in p.parts or "test" in p.parts)


def test_command_from_instruction(instr: str, ws: Path) -> list[str] | None:
    """If the task names the test command (`python3 test_x.py`, `python -m pytest tests/`, `pytest -q`), use it."""
    for cmd in re.findall(r"`([^`\n]{3,120})`", instr):
        parts = shlex.split(cmd)
        if not parts:
            continue
        if parts[0] in ("python", "python3", "py"):
            parts[0] = sys.executable
        elif parts[0] == "pytest":
            parts = [sys.executable, "-m", "pytest"] + parts[1:]
        else:
            continue
        if any(a.endswith(".py") or a in ("pytest", "unittest") or a.startswith("tests") for a in parts[1:]):
            return parts
    return None


def test_command(test_file: Path, ws: Path) -> list[str]:
    """unittest files run as scripts; pytest-style files (bare test_ functions) need a runner, otherwise
    `python test_x.py` exits 0 without running anything and a broken module would look fixed."""
    rel = str(test_file.relative_to(ws))
    src = read(test_file, 50_000)
    if "unittest" in src and "unittest.main" in src and test_file.parent == ws:
        return [sys.executable, rel]
    if "unittest" in src:
        return [sys.executable, "-m", "unittest", "-q", rel]
    return [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", rel]


def summarize_test_output(out: str, limit: int = 1600) -> str:
    """Keep the informative lines of a unittest/pytest report (failing test names, assertion lines, errors)."""
    keep = []
    for ln in out.splitlines():
        if re.match(r"\s*(FAIL|ERROR|FAILED|PASSED|Ran \d|E\s|>\s|\w*Error\b|\w*Exception\b|assert\b|\s+self\.assert|\s+File .*test|={3,}|_{3,}|\d+ (passed|failed))", ln):
            keep.append(ln.rstrip()[:200])
    text = "\n".join(keep) if keep else out
    return text[-limit:]


def run_tests(task: Task, cmd: list[str]) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, cwd=task.ws, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "tests timed out after 120s"
    except OSError as e:
        return False, f"cannot run tests: {e}"
    out = (p.stdout + p.stderr)
    if p.returncode == 5:  # pytest: no tests collected
        return False, out[-3000:] + "\n(no tests were collected)"
    return p.returncode == 0, out[-3000:]


def splice(source: str, patch: str) -> str:
    """Replace top-level defs/classes in source by those in patch; append new ones; keep new imports."""
    src_tree, patch_tree = ast.parse(source), ast.parse(patch)
    lines = source.splitlines(keepends=True)
    repl = {n.name: ast.get_source_segment(patch, n) for n in patch_tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    new_imports = [ast.get_source_segment(patch, n) for n in patch_tree.body
                   if isinstance(n, (ast.Import, ast.ImportFrom)) and ast.get_source_segment(patch, n) not in source]
    edits = []
    for n in src_tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in repl:
            start = (n.decorator_list[0].lineno if n.decorator_list else n.lineno) - 1
            edits.append((start, n.end_lineno, repl.pop(n.name) + "\n"))
    for start, end, text in sorted(edits, reverse=True):
        lines[start:end] = [text]
    out = "".join(lines)
    if new_imports:
        out = "\n".join(new_imports) + "\n" + out
    for text in repl.values():
        out += "\n\n" + text + "\n"
    return out


def split_patch(reply: str, srcs: list[Path], ws: Path) -> dict[Path, str]:
    """Map code blocks to source files: by '# file:' header, else by which file defines the patched names."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, re.S) or [extract_code(reply)]
    by_name = {}
    for p in srcs:
        try:
            for n in ast.parse(read(p)).body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    by_name.setdefault(n.name, p)
        except SyntaxError:
            pass
    out: dict[Path, list[str]] = {}
    for b in blocks:
        m = re.match(r"\s*#\s*file:\s*(\S+)", b)
        target = None
        if m:
            rel = m.group(1).strip("`'\"")
            target = next((p for p in srcs if str(p.relative_to(ws)) == rel or p.name == Path(rel).name), None)
            b = b[m.end():]
        if target is None:
            try:
                names = [n.name for n in ast.parse(b).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            except SyntaxError:
                names = []
            owners = collections.Counter(by_name[n] for n in names if n in by_name)
            target = owners.most_common(1)[0][0] if owners else srcs[0]
        out.setdefault(target, []).append(b)
    return {p: "\n\n".join(bs) for p, bs in out.items()}


async def solve_coding(task: Task, llm: LLM, models=None) -> str | None:
    models = models or (FAST, STRONG)
    pys = [p for p in ws_files(task.ws) if p.suffix == ".py"]
    tests = [p for p in pys if is_test_file(p)]
    srcs = [p for p in pys if p not in tests and p.name != "conftest.py" and p.stat().st_size > 0]
    if not tests or not srcs:
        return None
    m = re.search(r"\b(?:in|fix|repair)\s+(\S+\.py)", task.instruction, re.I)
    named = [p for p in srcs if m and p.name == m.group(1)]
    srcs = named + [p for p in srcs if p not in named]
    cmd = test_command_from_instruction(task.instruction, task.ws) or test_command(tests[0], task.ws)
    test_backup = {p: p.read_bytes() for p in tests}
    originals = {p: read(p) for p in srcs}

    ok, out = run_tests(task, cmd)
    if ok:
        return "DONE"
    instr = strip_boilerplate(task.instruction)
    shown_src = "".join(f"### {p.relative_to(task.ws)}\n```python\n{originals[p]}\n```\n\n" for p in srcs)[:14000]
    shown_tests = "".join(f"### {t.relative_to(task.ws)}\n```python\n{read(t, 6000)}\n```\n\n" for t in tests)[:10000]
    user = (f"Task: {instr}\n\n{shown_src}{shown_tests}### test output of `{' '.join(cmd[1:] if cmd[0] == sys.executable else cmd)}`\n```\n{summarize_test_output(out)}\n```\n\n"
            "Note: a function whose visible test passes can still be wrong (e.g. the test input happens to "
            "work). Check each function's logic against its name and include a fixed version of every "
            "function that is incorrect.")
    msgs = [{"role": "system", "content": SYS_CODE}, {"role": "user", "content": user}]
    attempts = list(models)
    for attempt, model in enumerate(attempts):
        reply = await llm.ask(msgs, model=model, max_tokens=1500 if "noreason" in model else 3000,
                              tag=f"{task.id}#{attempt}")
        if is_unsolvable(reply) and "```" not in reply:
            task.trace.append({"reply": reply[:300]})
            break
        if "```" not in reply and "def " not in reply:
            task.trace.append({"reply": reply[:300]})
            if len(attempts) < 3:
                attempts.append(model)  # one extra attempt only for a prose-only reply
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "You cannot explore or run commands. Reply now with the corrected code only, in ```python blocks."}]
            continue
        patches = split_patch(reply, srcs, task.ws)
        try:
            for p, patch in patches.items():
                p.write_text(splice(read(p), patch))
        except SyntaxError as e:
            out = f"Your code has a syntax error: {e}"
            ok = False
        else:
            for p, b in test_backup.items():
                p.write_bytes(b)
            ok, out = run_tests(task, cmd)
        task.trace.append({"patch": reply[:2000], "files": [str(p.relative_to(task.ws)) for p in patches], "ok": ok, "out": out[-800:]})
        if ok:
            return "DONE"
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": f"Tests still fail:\n```\n{summarize_test_output(out, 1200)}\n```\nIf the failing assertion contradicts "
                                             f"another test or the function's evident purpose, reply {UNSOLVABLE}; otherwise reply with the corrected code again."}]
    for p, src in originals.items():
        p.write_text(src)
    return None


SYS_KB = (
    "Answer a question about a knowledge base of files with a SHORT Python 3 snippet. Predefined (do not redefine, "
    "do not write files): kb {path: text}; records {title: {field: value}} parsed from headings/sections, "
    "'Key: value' lines and JSON objects (each also has '_file', '_text'); rec(name) -> record by title or the unique "
    "record whose text starts with/mentions the name; grep(regex) -> [(path, line)]; facts(name) -> sentences "
    "mentioning the name. rec() and ['field'] never raise: anything absent is MISSING and prints as UNSOLVABLE, so "
    "chain lookups without if/else, e.g. print(rec(rec('Ann Lee')['mentor'])['born']). Pass full names exactly as "
    "in the question; use only field names shown in the summary; a similar name is a different entity; in free text "
    "capture lazily, e.g. r'works at (.+?)(?= as | and |[.,;])'. Print ONLY the final answer (bare year/city/name/"
    f"number) on the last line, or {UNSOLVABLE} if the entity or the exact attribute asked for is absent (a birth "
    "city is not a birth year; never print '_text' or another field). One ```python block, 1-6 lines, no comments."
)


def kb_preview(ws: Path, question: str, limit: int = 2600) -> str:
    files = ws_files(ws)
    words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", question)}
    picked, seen = [], set()
    for p in files:  # one sample per (directory, filename prefix) so every record shape is shown
        key = (p.parent, re.split(r"[_\-.]", p.name)[0])
        if key not in seen:
            seen.add(key); picked.append(p)
    ent = 0
    for p in files:  # then up to two files named after entities in the question
        if any(w in str(p.relative_to(ws)).lower() for w in words) and p not in picked and ent < 2:
            picked.append(p); ent += 1
    parts, used = [], 0
    for p in picked[:5]:
        chunk = f"### {p.relative_to(ws)}\n{read(p, 700).strip()}"
        if used + len(chunk) > limit:
            break
        parts.append(chunk); used += len(chunk)
    dirs = sorted({str(p.parent.relative_to(ws)) for p in files})[:8]
    return (f"{len(files)} files in {dirs}; e.g. {', '.join(str(p.relative_to(ws)) for p in files[:3])}\n"
            f"Parsed summary: {kb_summary(ws)}\n\nSample file contents (truncated):\n" + "\n\n".join(parts))


async def solve_search_code(task: Task, llm: LLM, models=None) -> str | None:
    models = models or (FAST, STRONG)
    instr = strip_boilerplate(task.instruction)
    instr = re.sub(r"\s*Use the file tools.*?\.\s*", " ", instr).strip()
    user = f"Question: {instr}\n\n{kb_preview(task.ws, instr)}"
    msgs = [{"role": "system", "content": SYS_KB}, {"role": "user", "content": user}]
    max_tokens = 450
    for attempt, model in enumerate(models):
        reply, truncated = await llm.ask(msgs, model=model, tag=f"{task.id}#{attempt}", want_truncated=True,
                                         max_tokens=max_tokens if "noreason" in model else 2500)
        # the CHECK line is written before looking at the data, so it is not a verdict here: the code decides
        code = extract_code(reply)
        rc, out, err = run_python(code, task.ws, prelude=KB_PRELUDE)
        ans = clean_answer(out).strip(" .`*\"'")
        task.trace.append({"code": code, "rc": rc, "out": out[-300:], "err": err[-300:], "truncated": truncated})
        if rc == 0 and ans and not truncated:
            if is_unsolvable(ans):
                return None
            if len(ans) > 80:  # a whole record was printed: accept only if the question wants a number and the text has exactly one
                nums = re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w.])", ans)
                if NUMERIC_Q.search(instr) and len(nums) == 1:
                    task.trace.append({"extracted": nums[0], "from": ans[:120]})
                    ans = nums[0]
                else:
                    task.trace.append({"rejected": ans[:120]})
                    return None
            ans = refine_answer(instr, ans, task.ws)
            if not answer_type_ok(instr, ans):
                task.trace.append({"rejected": ans})
                return None
            return ans
        if truncated:
            max_tokens = 900
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "Your reply was cut off. Rewrite it much shorter (use records/rec()/grep()). One ```python block."}]
        else:
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"The program failed (exit {rc}).\nstdout:\n{out[-600:]}\nstderr:\n{err[-1000:]}\nFix it. One ```python block."}]
    return None


SEARCH_MODE = os.environ.get("SEARCH_MODE", "code")
SOLVERS = {"math": solve_compute, "data_analysis": solve_compute,
           "search": solve_search_code if SEARCH_MODE == "code" else solve_search, "coding": solve_coding}


# ---------------------------------------------------------------- planner

def load_tasks(root: Path) -> list[Task]:
    tasks = []
    for d in sorted(root.iterdir()):
        f = d / "task.json"
        if f.is_file():
            try:
                tasks.append(Task(d, json.loads(f.read_text())))
            except Exception as e:
                log.warning("skip %s: %s", d, e)
    return tasks


async def run(tasks: list[Task], ledger: Ledger, deadline: float, traces_out: Path | None):
    for t in tasks:
        try:
            t.est = estimate(t)
        except Exception:
            t.est = 2000
    # best reward per estimated token first
    queue = sorted(tasks, key=lambda t: t.reward / max(t.est, 1), reverse=True)
    global LLM_SLOTS
    LLM_SLOTS = asyncio.Semaphore(PARALLEL)
    sem = asyncio.Semaphore(min(PARALLEL + 1, 5))  # <=5 agent sessions at once (rules); calls <= PARALLEL
    stop = asyncio.Event()

    failures = {"streak": 0}

    async def work(t: Task):
        async with sem:
            while t.est > ledger.free and ledger.in_flight and not stop.is_set():
                await asyncio.sleep(0.3)  # money is reserved by calls in flight; wait for them to settle
            if stop.is_set() or time.time() > deadline:
                t.status = "skipped:time"
                return
            if t.est > ledger.free:
                t.status = "skipped:budget"
                return
            solver = SOLVERS.get(t.type, solve_compute)
            llm = LLM(ledger, t.trace)
            try:
                ans = await asyncio.wait_for(solver(t, llm), timeout=max(10, deadline - time.time()))
                if ans:
                    t.answer = ans
                    write_answer(t, ans)
                    t.status = "answered"
                else:
                    t.status = "declined"
            except BudgetExhausted as e:
                t.status = "skipped:budget"
                log.info("%s", e)
            except Exception as e:
                t.status = f"error:{type(e).__name__}"
                log.warning("%s failed: %r", t.id, e)
                failures["streak"] += 1
                if any(k in str(e).lower() for k in ("insufficient", "quota", "402", "budget")) \
                        or failures["streak"] >= 4:  # proxy stopped answering: stop burning time
                    stop.set()
            else:
                failures["streak"] = 0
            finally:
                spent = sum(x.get("cost", 0) for x in t.trace if "cost" in x)
                log.info("[%s] %s reward=%s est=%.0f spent=%.0f total=%.0f/%.0f free=%.0f reserved=%.0f",
                         t.status, t.id, t.reward, t.est, spent, ledger.spent, ledger.budget, ledger.free, ledger.reserved)
                if traces_out:
                    with open(traces_out, "a") as f:
                        f.write(json.dumps({"id": t.id, "type": t.type, "reward": t.reward, "est": t.est,
                                            "spent": spent, "status": t.status, "answer": t.answer,
                                            "expected": t.data.get("expected"), "trace": t.trace},
                                           ensure_ascii=False, default=str) + "\n")

    await asyncio.gather(*(work(t) for t in queue))
    if VERIFY:
        await verify_phase(tasks, ledger, deadline, sem, traces_out)


async def verify_phase(tasks, ledger: Ledger, deadline: float, sem: asyncio.Semaphore, traces_out):
    """Spend leftover budget on a cheap second opinion (the other fast model) for the highest-reward answered
    tasks. Agreement ends it; disagreement is settled by the reasoning model: majority wins, otherwise the
    reasoning answer. A None from a verifier is an abstention: verification never removes an answer."""
    cands = sorted((t for t in tasks if t.status == "answered" and t.type != "coding"),
                   key=lambda t: -t.reward)
    cands = cands[: max(1, int(round(len(cands) * VERIFY_TOP)))]
    floor = ledger.budget * VERIFY_KEEP

    async def check(t: Task):
        async with sem:
            if time.time() > deadline - 120 or ledger.free - 400 < floor:
                return
            solver = SOLVERS.get(t.type, solve_compute)
            trace: list = []
            llm = LLM(ledger, trace)
            verdict, final, a2 = "abstain", t.answer, None
            try:
                a2 = await asyncio.wait_for(solver(t, llm, models=(STRONG,)), timeout=max(10, deadline - time.time()))
                if a2 is None:
                    verdict = "abstain"
                elif same_answer(a2, t.answer):
                    verdict = "agree"
                elif ledger.free - 1500 < floor:
                    verdict = "disagree:unresolved"
                else:
                    a3 = await asyncio.wait_for(solver(t, llm, models=(REASON,)), timeout=max(10, deadline - time.time()))
                    if a3 is None:
                        verdict = "disagree:abstain"
                    elif same_answer(a3, t.answer):
                        verdict = "majority:first"
                    elif same_answer(a3, a2):
                        verdict, final = "majority:second", a2
                    else:
                        verdict, final = "reason", a3
                if final != t.answer:
                    write_answer(t, final)
                    t.answer = final
            except BudgetExhausted:
                verdict = "skipped:budget"
            except Exception as e:
                verdict = f"error:{type(e).__name__}"
                trace.append({"error": repr(e)[:300]})
            finally:
                trace.append({"verify": verdict, "a2": a2})
                spent = sum(x.get("cost", 0) for x in trace if "cost" in x)
                t.trace.extend(trace)
                log.info("[verify %s] %s reward=%s spent=%.0f total=%.0f/%.0f",
                         verdict, t.id, t.reward, spent, ledger.spent, ledger.budget)
                if traces_out:
                    with open(traces_out, "a") as f:
                        f.write(json.dumps({"id": t.id, "phase": "verify", "type": t.type, "reward": t.reward,
                                            "spent": spent, "status": t.status, "answer": t.answer,
                                            "expected": t.data.get("expected"), "trace": trace},
                                           ensure_ascii=False, default=str) + "\n")

    await asyncio.gather(*(check(t) for t in cands))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=os.environ.get("TASKS", "/data/tasks"))
    ap.add_argument("--traces", default=os.environ.get("TRACES"))
    ap.add_argument("--only", default=os.environ.get("ONLY"), help="dev: comma-separated task types to run")
    a = ap.parse_args()  # relative paths were given against the launch dir, we chdir'ed at import
    a.tasks_dir = os.path.join(_LAUNCH_DIR, a.tasks_dir)
    a.traces = a.traces and os.path.join(_LAUNCH_DIR, a.traces)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    for noisy in ("httpx", "openjiuwen", "openai", "common", "alembic", "llm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("llm").setLevel(logging.CRITICAL)  # openJiuwen echoes whole prompts on every error

    budget = float(os.environ.get("BUDGET", "100000"))
    ledger = Ledger(budget, margin=max(300, budget * 0.02))
    tasks = load_tasks(Path(a.tasks_dir))
    if a.only:
        tasks = [t for t in tasks if t.type in a.only.split(",")]
    log.info("tasks=%d budget=%.0f models=%s/%s", len(tasks), budget, FAST, STRONG)
    t0 = time.time()
    asyncio.run(run(tasks, ledger, t0 + WALL_LIMIT, Path(a.traces) if a.traces else None))
    summary = {s: sum(t.status == s for t in tasks) for s in sorted({t.status for t in tasks})}
    log.info("done in %.0fs: spent=%.0f reward=%s %s", time.time() - t0, ledger.spent,
             sum(t.reward for t in tasks if t.status == "answered"), summary)


if __name__ == "__main__":
    main()
