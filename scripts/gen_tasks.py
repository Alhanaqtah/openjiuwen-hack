"""Generate a synthetic task set in the organizers' format, with varied wording,
harder instances and ~20% unsolvable tasks (expected=null, check="unsolvable").

python scripts/gen_tasks.py OUT_DIR [--n 100] [--seed 1] [--unsolvable 0.2]
Coding tasks also get hidden_test_mylib.py in the task dir (not in workspace) for scoring.
"""
import argparse
import csv
import json
import random
import shutil
import textwrap
from pathlib import Path

FIRST = ["Alice", "Boris", "Chen", "Dmitri", "Elena", "Farid", "Grace", "Hiro", "Irina", "Jonas", "Kira",
         "Lena", "Maya", "Nikolai", "Olga", "Pavel", "Quinn", "Rosa", "Sergei", "Tanya", "Umar", "Vera"]
LAST = ["Novak", "Smith", "Wang", "Petrov", "Garcia", "Haddad", "Sato", "Kowalski", "Muller", "Ivanova"]
COMPANY = ["Granite AI", "Arktika Labs", "Fjord Data", "Bluefin Systems", "Juniper Foods", "Helix Motors",
           "Ember Bio", "Dune Robotics", "Cobalt Works", "Iris Optics", "Saffron Grid", "Tidal Works"]
CITY = ["Osaka", "Medan", "Tbilisi", "Hanoi", "Quito", "Izmir", "Lisbon", "Accra", "Oslo", "Lima", "Perth", "Riga"]
TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")


def rnd_name(r):
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


# ------------------------------------------------------------------ math
def gen_math(r, unsolvable):
    kind = r.choice(["warehouse", "account", "fleet", "percent", "library", "avg"])
    steps, val = [], 0
    if kind == "warehouse":
        val = r.randint(500, 5000)
        text = [f"A warehouse starts with {val} boxes."]
        for _ in range(r.randint(2, 6)):
            op = r.choice(["ship", "recv", "split", "double", "triple", "lose"])
            if op == "ship":
                d, n = r.randint(2, 9), r.randint(10, 90)
                text.append(f"It ships {n} boxes per day for {d} days."); val -= n * d
            elif op == "recv":
                k, n = r.randint(1, 5), r.randint(20, 150)
                text.append(f"It receives {k} deliveries of {n} boxes each."); val += k * n
            elif op == "split":
                k = r.randint(2, 4)
                text.append(f"Then the stock is split equally into {k} warehouses and it keeps one share (rounded down)."); val //= k
            elif op == "double":
                text.append("Then the stock is doubled by merging with a sister warehouse."); val *= 2
            elif op == "triple":
                text.append("Then two sister warehouses with the same stock merge into it."); val *= 3
            else:
                p = r.choice([10, 20, 25, 50])
                text.append(f"Then {p}% of the boxes are found damaged and discarded (round the discarded amount down)."); val -= (val * p) // 100
        if unsolvable:
            i = r.randint(1, len(text) - 1)
            text[i] = r.choice(["Then some boxes are shipped to a customer.", "Then a few deliveries arrive."])
        text.append("How many boxes does it have at the end?")
    elif kind == "account":
        val = r.randint(1000, 9000)
        text = [f"An account holds {val} credits."]
        for _ in range(r.randint(2, 5)):
            op = r.choice(["dep", "wd", "fee", "int"])
            if op == "dep":
                m, n = r.randint(2, 6), r.randint(50, 400)
                text.append(f"{n} credits are deposited every month for {m} months."); val += m * n
            elif op == "wd":
                n = r.randint(100, 900); text.append(f"A withdrawal of {n} credits is made."); val -= n
            elif op == "fee":
                n = r.randint(5, 40); k = r.randint(2, 7)
                text.append(f"A fee of {n} credits is charged {k} times."); val -= n * k
            else:
                p = r.choice([5, 10, 20]); text.append(f"Then {p}% interest is added (rounded down)."); val += (val * p) // 100
        if unsolvable:
            text.insert(r.randint(1, len(text)), "Later an unknown amount is transferred out.")
        text.append("What is the final balance?")
    elif kind == "fleet":
        buses, cap = r.randint(3, 12), r.randint(20, 60)
        trips = r.randint(2, 8)
        people = r.randint(100, 2000)
        text = [f"A company has {buses} buses, each carrying at most {cap} passengers per trip.",
                f"Each bus makes {trips} trips a day.", f"{people} people need to travel today."]
        if unsolvable:
            text[1] = "Each bus makes several trips a day."
        q = r.choice(["cap", "left"])
        if q == "cap":
            text.append("How many passengers can be carried in total per day?"); val = buses * cap * trips
        else:
            text.append("How many people cannot be carried today (0 if everyone fits)?"); val = max(0, people - buses * cap * trips)
    elif kind == "percent":
        price, n = r.randint(20, 400), r.randint(3, 40)
        d = r.choice([10, 15, 20, 25, 30])
        text = [f"A shop sells {n} items at {price} each.", f"Every item is discounted by {d}%."]
        if unsolvable:
            text[1] = "Every item is discounted by a percentage announced in the store."
        text.append("What is the total revenue after the discount? Round down to a whole number.")
        val = (n * price * (100 - d)) // 100
    elif kind == "library":
        books = r.randint(200, 3000)
        text = [f"A library owns {books} books."]
        out = r.randint(10, 150); text.append(f"{out} books are currently borrowed.")
        ret = r.randint(1, out); text.append(f"{ret} of the borrowed books are returned today.")
        if unsolvable:
            text[1] = "Some books are currently borrowed."
        text.append("How many books are on the shelves at the end of the day (borrowed books are not on the shelves)?")
        val = books - out + ret
    else:
        xs = [r.randint(10, 99) for _ in range(r.randint(4, 9))]
        text = [f"Test scores of a group are: {', '.join(map(str, xs))}."]
        q = r.choice(["sum", "avg", "above"])
        if unsolvable:
            text[0] = f"Test scores of a group are: {', '.join(map(str, xs[:-1]))} and one more score that was not recorded."
        if q == "sum":
            text.append("What is the total of all scores?"); val = sum(xs)
        elif q == "avg":
            text.append("What is the average score, rounded down to an integer?"); val = sum(xs) // len(xs)
        else:
            t = r.randint(30, 80); text.append(f"How many scores are strictly greater than {t}?"); val = sum(x > t for x in xs)
    instr = " ".join(text) + r.choice([" Compute it with the bash tool (python3 -c).", " Use python3 via bash to compute the result.", ""]) + TAIL
    return instr, None if unsolvable else str(val), "exact_int", {}


# ------------------------------------------------------------------ data analysis
def gen_data(r, unsolvable):
    schema = r.choice(["sales", "employees", "weather"])
    n = r.choice([20, 60, 300, 1500])
    rows = []
    if schema == "sales":
        header = ["order_id", "region", "product", "month", "units", "price"]
        prods = ["widget", "gadget", "sprocket", "gizmo"]
        regs = ["north", "south", "east", "west"]
        for i in range(1, n + 1):
            rows.append([i, r.choice(regs), r.choice(prods), r.randint(1, 12), r.randint(1, 60), r.randint(5, 120)])
        p, m1 = r.choice(prods), r.randint(1, 10); m2 = min(12, m1 + r.randint(0, 3))
        reg = r.choice(regs)
        q = r.choice(["units", "revenue", "orders", "avg", "maxregion"])
        sel = [x for x in rows if x[2] == p and m1 <= x[3] <= m2]
        if q == "units":
            qs = f"How many units of product '{p}' were sold in months {m1} through {m2} inclusive?"; val = sum(x[4] for x in sel)
        elif q == "revenue":
            qs = f"What is the total revenue (units times price) for product '{p}' in region '{reg}'?"; val = sum(x[4] * x[5] for x in rows if x[2] == p and x[1] == reg)
        elif q == "orders":
            qs = f"How many orders of '{p}' have more than 30 units?"; val = sum(1 for x in rows if x[2] == p and x[4] > 30)
        elif q == "avg":
            sel2 = [x for x in rows if x[1] == reg]
            qs = f"What is the average number of units per order in region '{reg}', rounded down to an integer?"; val = sum(x[4] for x in sel2) // max(1, len(sel2))
        else:
            tot = {g: 0 for g in regs}
            for x in rows:
                if x[2] == p: tot[x[1]] += x[4]
            best = max(tot, key=lambda g: (tot[g], g))
            qs = f"Which region sold the most units of '{p}'? Answer with the region name."; val = best
        if unsolvable:
            qs = r.choice([f"What was the total discount given on product '{p}'?",
                           f"How many units of '{p}' were returned by customers?",
                           f"How many units of '{p}' were sold in month 14?".replace("14", "0") if False else f"What was the profit margin on '{p}' orders in region '{reg}'?"])
    elif schema == "employees":
        header = ["emp_id", "department", "city", "salary", "years"]
        deps = ["sales", "engineering", "support", "finance"]
        cities = ["Oslo", "Lima", "Perth", "Riga"]
        for i in range(1, n + 1):
            rows.append([i, r.choice(deps), r.choice(cities), r.randint(30, 150) * 100, r.randint(0, 20)])
        d, c = r.choice(deps), r.choice(cities)
        q = r.choice(["count", "payroll", "senior", "maxsal", "avg"])
        if q == "count":
            qs = f"How many employees work in the {d} department in {c}?"; val = sum(1 for x in rows if x[1] == d and x[2] == c)
        elif q == "payroll":
            qs = f"What is the total salary of the {d} department?"; val = sum(x[3] for x in rows if x[1] == d)
        elif q == "senior":
            y = r.randint(5, 15); qs = f"How many employees have at least {y} years of experience?"; val = sum(1 for x in rows if x[4] >= y)
        elif q == "maxsal":
            qs = f"What is the highest salary in {c}?"; val = max([x[3] for x in rows if x[2] == c] or [0])
        else:
            sel = [x for x in rows if x[1] == d]
            qs = f"What is the average salary in the {d} department, rounded down?"; val = sum(x[3] for x in sel) // max(1, len(sel))
        if unsolvable:
            qs = r.choice([f"How many employees in {d} are older than 40?", f"What is the average bonus paid in {c}?",
                           f"How many employees in {d} joined in 2021?"])
    else:
        header = ["day", "station", "temp_c", "rain_mm"]
        st = ["A", "B", "C"]
        for i in range(1, n + 1):
            rows.append([i, r.choice(st), r.randint(-10, 35), r.choice([0, 0, 0, r.randint(1, 40)])])
        s = r.choice(st)
        q = r.choice(["rainy", "hot", "maxt", "sumrain"])
        if q == "rainy":
            qs = f"On how many days did station {s} record any rain (rain_mm > 0)?"; val = sum(1 for x in rows if x[1] == s and x[3] > 0)
        elif q == "hot":
            t = r.randint(20, 30); qs = f"How many records have temp_c strictly above {t}?"; val = sum(1 for x in rows if x[2] > t)
        elif q == "maxt":
            qs = f"What is the maximum temp_c at station {s}?"; val = max([x[2] for x in rows if x[1] == s] or [0])
        else:
            qs = f"What is the total rain_mm at station {s}?"; val = sum(x[3] for x in rows if x[1] == s)
        if unsolvable:
            qs = r.choice([f"What was the average humidity at station {s}?", f"What was the wind speed on the rainiest day at station {s}?"])
    fname = r.choice(["data.csv", f"{schema}.csv"])
    cols = ",".join(header)
    instr = r.choice([f"The file {fname} has columns {cols} ({n} rows). {qs}",
                      f"Analyse {fname} (columns: {cols}; {n} data rows). {qs}",
                      f"Using the CSV file {fname} with columns {cols}, answer: {qs}"]) \
        + r.choice([" Use bash with python3 to compute it.", " Compute the answer with python3.", ""]) + TAIL
    check = "exact_str" if isinstance(val, str) else "exact_int"
    return instr, None if unsolvable else str(val), check, {fname: (header, rows)}


# ------------------------------------------------------------------ search
def gen_search(r, unsolvable):
    npeople = r.choice([20, 40, 80])
    comps = r.sample(COMPANY, r.randint(6, 10))
    cities = r.sample(CITY, len(comps))
    people = {}
    names = set()
    while len(names) < npeople:
        names.add(rnd_name(r))
    names = sorted(names)
    for nm in names:
        people[nm] = {"Born": r.randint(1955, 2004), "Employer": r.choice(comps), "Mentor": r.choice(names)}
    comp_info = {c: {"Headquarters": cities[i], "Founded": r.randint(1950, 2020)} for i, c in enumerate(comps)}
    files = {}
    for nm, d in people.items():
        files[f"kb/person_{nm.lower().replace(' ', '_')}.md"] = f"# {nm}\n\n" + "\n".join(f"{k}: {v}" for k, v in d.items()) + "\n"
    for c, d in comp_info.items():
        files[f"kb/company_{c.lower().replace(' ', '_')}.md"] = f"# {c}\n\n" + "\n".join(f"{k}: {v}" for k, v in d.items()) + "\n"
    p = r.choice(names)
    hop = r.choice(["born", "hq", "founded", "mentor_born", "mentor_hq", "colleague_count"])
    if hop == "born":
        q, val = f"In which year was {p} born?", people[p]["Born"]
    elif hop == "hq":
        q, val = f"In which city is the headquarters of {p}'s employer?", comp_info[people[p]["Employer"]]["Headquarters"]
    elif hop == "founded":
        q, val = f"In what year was the company that employs {p} founded?", comp_info[people[p]["Employer"]]["Founded"]
    elif hop == "mentor_born":
        q, val = f"In which year was {p}'s mentor born?", people[people[p]["Mentor"]]["Born"]
    elif hop == "mentor_hq":
        q, val = f"Where (city) is the headquarters of the company where {p}'s mentor works?", comp_info[people[people[p]["Mentor"]]["Employer"]]["Headquarters"]
    else:
        c = people[p]["Employer"]
        q, val = f"How many people in the knowledge base work for {c}?", sum(1 for d in people.values() if d["Employer"] == c)
    if unsolvable:
        mode = r.choice(["missing_person", "missing_field", "dangling"])
        if mode == "missing_person":
            ghost = rnd_name(r)
            while ghost in people:
                ghost = rnd_name(r)
            q = r.choice([f"In which year was {ghost} born?", f"In which city is the headquarters of {ghost}'s employer?"])
        elif mode == "missing_field":
            q = r.choice([f"What is {p}'s email address?", f"How many employees does {people[p]['Employer']} have?",
                          f"In which city was {p} born?"])
        else:
            # employer referenced but company file missing
            c = people[p]["Employer"]
            files.pop(f"kb/company_{c.lower().replace(' ', '_')}.md", None)
            q = f"In which city is the headquarters of {p}'s employer?"
    instr = r.choice([f"The directory kb/ contains a small knowledge base of markdown files. {q} Use the file tools or bash (ls, grep, cat) to find it.",
                      f"Look through the markdown notes in kb/ and answer: {q}",
                      f"{q} The answer is somewhere in the knowledge base under kb/ (markdown files)."]) + TAIL
    check = "exact_int" if isinstance(val, int) else "exact_str"
    return instr, None if unsolvable else str(val), check, files


# ------------------------------------------------------------------ coding
FUNCS = {
    "total": ("def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
              ["def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n",
               "def total(xs):\n    s = 0\n    for x in xs[1:]:\n        s += x\n    return s\n"],
              [("[1, 2, 3]", "6"), ("[]", "0"), ("[5]", "5"), ("[-1, 1, 10]", "10")]),
    "maximum": ("def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
                ["def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
                 "def maximum(xs):\n    best = 0\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n"],
                [("[3, 9, 2]", "9"), ("[-5, -2, -9]", "-2"), ("[7]", "7")]),
    "reverse_words": ("def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
                      ["def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n",
                       "def reverse_words(s):\n    return s[::-1]\n"],
                      [("'c a b'", "'b a c'"), ("'hello'", "'hello'"), ("'one two three'", "'three two one'")]),
    "count_even": ("def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
                   ["def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
                    "def count_even(xs):\n    return len([x for x in xs if x // 2 == 0])\n"],
                   [("[1, 2, 4, 6, 7]", "3"), ("[]", "0"), ("[0, -2, 3]", "2")]),
    "average": ("def average(xs):\n    return sum(xs) / len(xs)\n",
                ["def average(xs):\n    return sum(xs) / (len(xs) + 1)\n", "def average(xs):\n    return sum(xs) // len(xs)\n"],
                [("[2, 4, 6]", "4"), ("[1, 2]", "1.5"), ("[10]", "10")]),
    "first_n": ("def first_n(xs, n):\n    return xs[:n]\n",
                ["def first_n(xs, n):\n    return xs[:n - 1]\n", "def first_n(xs, n):\n    return xs[n:]\n"],
                [("[5, 6, 7, 8], 3", "[5, 6, 7]"), ("[1, 2], 5", "[1, 2]"), ("[1, 2, 3], 0", "[]")]),
    "is_palindrome": ("def is_palindrome(s):\n    t = ''.join(ch.lower() for ch in s if ch.isalnum())\n    return t == t[::-1]\n",
                      ["def is_palindrome(s):\n    return s == s[::-1]\n",
                       "def is_palindrome(s):\n    t = ''.join(ch for ch in s if ch.isalnum())\n    return t == t[::-1]\n"],
                      [("'A man, a plan, a canal: Panama'", "True"), ("'abc'", "False"), ("'Noon'", "True")]),
    "fizzbuzz": ("def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)\n",
                 ["def fizzbuzz(n):\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    return str(n)\n",
                  "def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return n\n"],
                 [("15", "'FizzBuzz'"), ("9", "'Fizz'"), ("10", "'Buzz'"), ("7", "'7'")]),
    "word_freq": ("def word_freq(s):\n    d = {}\n    for w in s.lower().split():\n        d[w] = d.get(w, 0) + 1\n    return d\n",
                  ["def word_freq(s):\n    d = {}\n    for w in s.split():\n        d[w] = d.get(w, 0) + 1\n    return d\n",
                   "def word_freq(s):\n    d = {}\n    for w in s.lower().split():\n        d[w] = 1\n    return d\n"],
                  [("'a A b'", "{'a': 2, 'b': 1}"), ("''", "{}"), ("'x x x'", "{'x': 3}")]),
    "clamp": ("def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
              ["def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n", "def clamp(x, lo, hi):\n    return max(lo, min(x, hi - 1))\n"],
              [("5, 0, 10", "5"), ("-3, 0, 10", "0"), ("42, 0, 10", "10")]),
    "flatten": ("def flatten(xss):\n    out = []\n    for xs in xss:\n        out.extend(xs)\n    return out\n",
                ["def flatten(xss):\n    out = []\n    for xs in xss:\n        out.append(xs)\n    return out\n",
                 "def flatten(xss):\n    out = []\n    for xs in xss[:-1]:\n        out.extend(xs)\n    return out\n"],
                [("[[1, 2], [3]]", "[1, 2, 3]"), ("[]", "[]"), ("[[], [4, 5]]", "[4, 5]")]),
    "second_largest": ("def second_largest(xs):\n    u = sorted(set(xs))\n    return u[-2] if len(u) >= 2 else None\n",
                       ["def second_largest(xs):\n    u = sorted(xs)\n    return u[-2]\n",
                        "def second_largest(xs):\n    u = sorted(set(xs))\n    return u[1]\n"],
                       [("[3, 9, 9, 2]", "3"), ("[1, 2, 3]", "2"), ("[4, 4, 1]", "1")]),
}


def gen_coding(r, unsolvable):
    names = r.sample(list(FUNCS), r.randint(4, 7))
    buggy = set(r.sample(names, r.randint(1, min(3, len(names)))))
    src, vis, hid = [], [], []
    for nm in names:
        good, bugs, cases = FUNCS[nm]
        src.append(r.choice(bugs) if nm in buggy else good)
        c0 = cases[0]
        vis.append((nm, c0))
        hid.extend((nm, c) for c in cases)
    if unsolvable:
        # two visible tests demand different results for the same input
        nm, (args, res) = r.choice(vis)
        vis.append((nm + "_again", (args, res + " + 1" if res.lstrip("-").isdigit() else "'impossible'")))
    mod = r.choice(["mylib", "utils", "core"])
    testname = f"test_{mod}.py"

    def render(cases):
        body = "".join(f"    def test_{nm if not nm.endswith('_again') else nm}(self):\n"
                       f"        self.assertEqual(m.{nm.replace('_again', '')}({a}), {res})\n\n" for nm, (a, res) in cases)
        return f"import unittest\n\nimport {mod} as m\n\n\nclass T(unittest.TestCase):\n{body}\nif __name__ == '__main__':\n    unittest.main()\n"
    files = {f"{mod}.py": "\n\n".join(src), testname: render(vis)}
    hidden = render(hid)
    instr = r.choice([f"Run `python3 {testname}` in the current directory; some tests fail. Fix the bugs in {mod}.py (do not edit the tests) until all tests pass.",
                      f"The unit tests in {testname} currently fail. Repair {mod}.py so that the whole suite passes; the tests themselves must stay untouched.",
                      f"{mod}.py contains one or more bugs that make `python3 {testname}` fail. Fix the implementation (not the tests)."]) \
        + " Work only inside the current directory. When you are done, write ONLY the word DONE to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
    return instr, None if unsolvable else "tests_pass", "unittest", files, hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--unsolvable", type=float, default=0.2)
    a = ap.parse_args()
    r = random.Random(a.seed)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    types = ["math", "data_analysis", "search", "coding"]
    for i in range(a.n):
        t = types[i % 4]
        uns = r.random() < a.unsolvable
        d = out / f"{t}_{i:04d}"
        ws = d / "workspace"
        ws.mkdir(parents=True)
        hidden = None
        if t == "math":
            instr, exp, chk, files = gen_math(r, uns)
        elif t == "data_analysis":
            instr, exp, chk, files = gen_data(r, uns)
            for fname, (header, rows) in files.items():
                with open(ws / fname, "w", newline="") as f:
                    w = csv.writer(f); w.writerow(header); w.writerows(rows)
            files = {}
        elif t == "search":
            instr, exp, chk, files = gen_search(r, uns)
        else:
            instr, exp, chk, files, hidden = gen_coding(r, uns)
        for fname, content in files.items():
            p = ws / fname
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        if hidden:
            (d / "hidden_tests.py").write_text(hidden)
        task = {"task_id": d.name, "task_type": t, "instruction": instr, "reward": r.randint(1, 5),
                "expected": exp, "check": "unsolvable" if uns else chk}
        (d / "task.json").write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n")
    print(f"{a.n} tasks -> {out}")


if __name__ == "__main__":
    main()
