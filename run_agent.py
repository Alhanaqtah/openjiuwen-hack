"""Budget-aware harness for openJiuwen Challenge 2026.

python run_agent.py --tasks-dir /data/tasks
env: LLM_BASE_URL, LLM_API_KEY, BUDGET (output-token equivalents; input counts 1/10)
"""
import argparse
import ast
import asyncio
import json
import logging
import os
import re
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

from openjiuwen.core.foundation.llm.model import Model  # noqa: E402
from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig, ModelRequestConfig  # noqa: E402

log = logging.getLogger("harness")

FAST = os.environ.get("MODEL_FAST", "qwen3.8-27b-noreason")      # primary model
STRONG = os.environ.get("MODEL_STRONG", "qwen3.6-35b-a3b-noreason")  # second opinion / retry
REASON = os.environ.get("MODEL_REASON", "qwen3.8-27b")  # tie-breaker only (reasoning tokens are expensive)
VERIFY = os.environ.get("VERIFY", "1") == "1"  # phase 2: second opinion on answered tasks while budget remains
VERIFY_KEEP = float(os.environ.get("VERIFY_KEEP", "0.35"))  # share of budget never spent on verification
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
kb = {}
for _root, _dirs, _files in os.walk("."):
    for _f in sorted(_files):
        _p = os.path.join(_root, _f)[2:]
        if _f.startswith(".") or _f in ("task.json", "answer.txt") or os.path.getsize(_p) > 2_000_000:
            continue
        try:
            kb[_p] = open(_p, encoding="utf-8", errors="replace").read()
        except OSError:
            pass
records = {}
for _p, _t in kb.items():
    _m = re.search(r"^#\s*(.+?)\s*$", _t, re.M)
    _title = _m.group(1).strip() if _m else os.path.splitext(os.path.basename(_p))[0]
    _fields = {}
    for _k, _v in re.findall(r"^\**([A-Za-z][A-Za-z0-9 _/-]{0,40}?)\**\s*:\s*(.+?)\s*$", _t, re.M):
        _fields.setdefault(_k.strip().lower(), _v.strip())
    _fields["_file"] = _p
    records[_title] = _fields
def rec(name):
    """record by title, case-insensitive exact match; None if absent"""
    if name is None: return None
    n = str(name).strip().lower()
    for _k, _v in records.items():
        if _k.lower() == n: return _v
    return None
'''


def remove_answer(task: Task):
    for target in (task.dir / "answer.txt", task.ws / "answer.txt"):
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning("%s: cannot remove %s: %s", task.id, target, e)


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
    return bool(re.search(r"\b(not given|not specified|not provided|not present|not available|unspecified|missing|absent|unknown|no such|does not exist|cannot|unsolvable)\b", line, re.I)) \
        and not re.search(r"\b(all (?:are |quantities are )?(?:given|present|provided|specified)|nothing (?:is )?missing|none missing)\b", line, re.I)


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
        py = sum(p.stat().st_size for p in ws_files(task.ws) if p.suffix == ".py")
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
    "You solve quantitative tasks by writing a short Python 3 program (stdlib only; csv module for CSV files). "
    "The program runs in the task directory and must print ONLY the final answer on the last line. "
    "Rules: follow the wording literally and in order (inclusive ranges; 'added'/'received' means +, 'charged'/"
    "'shipped' means -; 'rounded down' means // or int()); keep integer arithmetic for whole items; "
    "use only columns/fields that actually exist, never reinterpret a column as a different quantity "
    "(years of experience are not age, salary is not bonus, units are not returns). "
    f"If a needed quantity, column or value is missing or unspecified, the program must print {UNSOLVABLE} "
    "instead of guessing or assuming 0. Reply format: first a single line 'CHECK: <which quantities the "
    "question needs and whether each is given>' then one ```python block with short code, no comments."
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
    "You fix bugs in a Python module so that its unit tests pass. The grader runs HIDDEN tests that are broader "
    "than the visible ones (empty inputs, single elements, negatives, duplicates, mixed case, boundaries), so "
    "review EVERY function, not only the failing ones, and make each behave as its name and docstring imply. "
    "Never touch the tests. Reply with ONE ```python block containing ONLY the corrected top-level functions/"
    "classes (complete definitions); functions that are already correct may be omitted. If two tests demand "
    f"contradictory results for the same input, reply {UNSOLVABLE} and no code."
)


def run_tests(task: Task, test_file: Path) -> tuple[bool, str]:
    try:
        p = subprocess.run([sys.executable, test_file.name], cwd=task.ws, capture_output=True,
                           text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "tests timed out after 120s"
    out = (p.stdout + p.stderr)
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


async def solve_coding(task: Task, llm: LLM, models=None) -> str | None:
    models = models or (FAST, STRONG)
    tests = [p for p in ws_files(task.ws) if p.name.startswith("test") and p.suffix == ".py"]
    srcs = [p for p in ws_files(task.ws) if p.suffix == ".py" and p not in tests]
    if not tests or not srcs:
        return None
    test_file = tests[0]
    m = re.search(r"Fix the bugs in (\S+\.py)", task.instruction)
    target = next((p for p in srcs if m and p.name == m.group(1)), srcs[0])
    test_backup = {p: p.read_bytes() for p in tests}
    original = read(target)

    ok, out = run_tests(task, test_file)
    if ok:
        return "DONE"
    instr = strip_boilerplate(task.instruction)
    user = (f"Task: {instr}\n\n### {target.name}\n```python\n{original}\n```\n\n"
            + "".join(f"### {t.name}\n```python\n{read(t)}\n```\n\n" for t in tests)
            + f"### test output\n```\n{out[-2500:]}\n```\n\n"
            "Note: a function whose visible test passes can still be wrong (e.g. the test input happens to "
            "work). Check each function's logic against its name and include a fixed version of every "
            "function that is incorrect.")
    msgs = [{"role": "system", "content": SYS_CODE}, {"role": "user", "content": user}]
    for attempt, model in enumerate(models):
        reply = await llm.ask(msgs, model=model, max_tokens=1500 if "noreason" in model else 3000,
                              tag=f"{task.id}#{attempt}")
        if is_unsolvable(reply) and "```" not in reply:
            task.trace.append({"reply": reply})
            break
        patch = extract_code(reply)
        try:
            target.write_text(splice(read(target), patch))
        except SyntaxError as e:
            out = f"Your code has a syntax error: {e}"
        else:
            for p, b in test_backup.items():
                p.write_bytes(b)
            ok, out = run_tests(task, test_file)
        task.trace.append({"patch": patch, "ok": ok, "out": out[-800:]})
        if ok:
            return "DONE"
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": f"Tests still fail:\n```\n{out[-2000:]}\n```\nReply with the corrected functions again."}]
    target.write_text(original)
    return None


SYS_KB = (
    "You answer a question about a small knowledge base of text files by writing a SHORT Python 3 snippet "
    "(at most ~15 lines). These variables are already defined for you, do not redefine them:\n"
    "  kb: dict[path -> file text] for every file under the current directory;\n"
    "  records: dict[title -> {field_lower: value}] parsed from '# Title' headings and 'Key: Value' lines "
    "(fields are lower-case keys, e.g. records['Alice Smith']['employer']); '_file' holds the path;\n"
    "  rec(name) -> that record by title (case-insensitive exact match) or None.\n"
    "Follow references between records when needed (person -> mentor/employer -> company). Match names "
    "exactly; a similar name is a different entity. Use field names exactly as they appear in the sample "
    "files (lower-cased). Print ONLY the final answer on the last line (bare "
    f"year, city, name or number). If an entity, file or field needed for the answer is absent, print {UNSOLVABLE} "
    "instead of guessing; answer only the field asked for (a birth city is not a birth year). Do not hard-code "
    "the answer. Reply format: one line 'CHECK: <entities/fields needed, present or not>' then one ```python "
    "block with short code, no comments."
)


def kb_preview(ws: Path, question: str, limit: int = 2500) -> str:
    files = ws_files(ws)
    words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", question)}
    names = sorted({re.split(r"[_\-.]", p.name)[0] for p in files})
    picked, seen = [], set()
    for p in files:  # one sample per filename prefix (person_, company_, ...) so every record shape is shown
        pre = re.split(r"[_\-.]", p.name)[0]
        if pre not in seen:
            seen.add(pre); picked.append(p)
    for p in files:  # then files named after entities in the question
        if any(w in p.name.lower() for w in words) and p not in picked:
            picked.append(p)
    parts, used = [], 0
    for p in picked[:5]:
        chunk = f"### {p.relative_to(ws)}\n{read(p, 800).strip()}"
        if used + len(chunk) > limit:
            break
        parts.append(chunk); used += len(chunk)
    return (f"{len(files)} files; filename prefixes: {', '.join(names[:12])}; e.g. "
            f"{', '.join(str(p.relative_to(ws)) for p in files[:3])}\n\nSample files:\n" + "\n\n".join(parts))


async def solve_search_code(task: Task, llm: LLM, models=None) -> str | None:
    models = models or (FAST, STRONG)
    instr = strip_boilerplate(task.instruction)
    instr = re.sub(r"\s*Use the file tools.*?\.\s*", " ", instr).strip()
    user = f"Question: {instr}\n\n{kb_preview(task.ws, instr)}"
    msgs = [{"role": "system", "content": SYS_KB}, {"role": "user", "content": user}]
    max_tokens = 400
    for attempt, model in enumerate(models):
        reply, truncated = await llm.ask(msgs, model=model, tag=f"{task.id}#{attempt}", want_truncated=True,
                                         max_tokens=max_tokens if "noreason" in model else 2500)
        if check_line_says_unsolvable(reply) and not truncated:
            task.trace.append({"check": reply[:300]})
            return None
        code = extract_code(reply)
        rc, out, err = run_python(code, task.ws, prelude=KB_PRELUDE)
        ans = clean_answer(out).strip(" .`*\"'")
        task.trace.append({"code": code, "rc": rc, "out": out[-300:], "err": err[-300:], "truncated": truncated})
        if rc == 0 and ans and not truncated:
            return None if is_unsolvable(ans) or len(ans) > 80 else ans
        if truncated:
            max_tokens = 900
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "Your reply was cut off. Rewrite it much shorter (use records/rec()). One ```python block."}]
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
    """Spend leftover budget on a second opinion from the reasoning model for answered tasks, highest reward
    first. Disagreement -> third vote (other fast model); majority wins; without a majority the reasoning
    answer is kept. A None from a verifier is an abstention: verification never removes an answer."""
    cands = sorted((t for t in tasks if t.status == "answered" and t.type != "coding"),
                   key=lambda t: -t.reward)
    floor = ledger.budget * VERIFY_KEEP

    async def check(t: Task):
        async with sem:
            if time.time() > deadline - 120 or ledger.free - 900 < floor:
                return
            solver = SOLVERS.get(t.type, solve_compute)
            trace: list = []
            llm = LLM(ledger, trace)
            verdict, final, a2 = "abstain", t.answer, None
            try:
                a2 = await asyncio.wait_for(solver(t, llm, models=(REASON,)), timeout=max(10, deadline - time.time()))
                if a2 is None:
                    verdict = "abstain"
                elif same_answer(a2, t.answer):
                    verdict = "agree"
                elif ledger.free - 400 < floor:
                    verdict = "disagree:unresolved"
                else:
                    a3 = await asyncio.wait_for(solver(t, llm, models=(STRONG,)), timeout=max(10, deadline - time.time()))
                    if same_answer(a3, t.answer):
                        verdict = "majority:first"
                    elif same_answer(a3, a2):
                        verdict, final = "majority:reason", a2
                    else:
                        verdict, final = "reason", a2
                if final != t.answer:
                    write_answer(t, final)
                    t.answer = final
            except BudgetExhausted:
                verdict = "skipped:budget"
            except Exception as e:
                verdict = f"error:{type(e).__name__}"
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
    for noisy in ("httpx", "openjiuwen", "openai", "common"):
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
