"""Tasks in formats that differ from the organizers' examples, to check the harness is not tuned to them:
prose/JSON/single-file knowledge bases, TSV/JSON/semicolon tables, multi-file & class-based coding, other wording.

python scripts/gen_alt.py OUT_DIR [--seed 3]
"""
import argparse
import json
import random
import shutil
from pathlib import Path

TAIL = " When finished, write ONLY the final answer into answer.txt in the current directory."
FIRST = ["Alice", "Boris", "Chen", "Dmitri", "Elena", "Farid", "Grace", "Hiro", "Irina", "Jonas", "Kira", "Lena"]
LAST = ["Novak", "Smith", "Wang", "Petrov", "Garcia", "Haddad", "Sato", "Kowalski"]
COMPANY = ["Granite AI", "Arktika Labs", "Fjord Data", "Bluefin Systems", "Juniper Foods", "Helix Motors"]
CITY = ["Osaka", "Medan", "Tbilisi", "Hanoi", "Quito", "Izmir"]


def people(r, n):
    names = set()
    while len(names) < n:
        names.add(f"{r.choice(FIRST)} {r.choice(LAST)}")
    names = sorted(names)
    comps = {c: {"city": CITY[i], "founded": r.randint(1950, 2015)} for i, c in enumerate(COMPANY)}
    ppl = {nm: {"born": r.randint(1955, 2000), "employer": r.choice(COMPANY), "mentor": r.choice(names),
                "role": r.choice(["engineer", "analyst", "designer", "manager"])} for nm in names}
    return names, ppl, comps


def write(d, files):
    for rel, content in files.items():
        p = d / "workspace" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def task(d, ttype, instr, expected, check):
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.json").write_text(json.dumps({"task_id": d.name, "task_type": ttype, "instruction": instr,
                                             "reward": 3, "expected": expected, "check": check}, indent=2) + "\n")


def gen(out, r):
    i = 0
    def nxt(t):
        nonlocal i; i += 1; return out / f"{t}_alt{i:03d}"

    # ---- search: prose notes in nested folders, .txt
    names, ppl, comps = people(r, 25)
    files = {}
    for nm, p in ppl.items():
        files[f"notes/people/{nm.split()[1].lower()}/{nm.split()[0].lower()}.txt"] = (
            f"{nm} was born in {p['born']}. {nm.split()[0]} works at {p['employer']} as a {p['role']} "
            f"and was mentored by {p['mentor']}.\n")
    for c, info in comps.items():
        files[f"notes/orgs/{c.lower().replace(' ', '-')}.txt"] = f"{c} is headquartered in {info['city']}; it was founded in {info['founded']}.\n"
    pick = r.choice(names)
    d = nxt("search"); write(d, files)
    task(d, "search", f"Somewhere under notes/ there are plain-text notes about people and organisations. Which city is the headquarters of the organisation where {pick} works?" + TAIL, comps[ppl[pick]["employer"]]["city"], "exact_str")
    pick = r.choice(names)
    d = nxt("search"); write(d, files)
    task(d, "search", f"Using the notes in notes/, find the birth year of the person who mentored {pick}." + TAIL, str(ppl[ppl[pick]["mentor"]]["born"]), "exact_int")
    d = nxt("search"); write(d, files)
    ghost = "Olga Ivanova"
    task(d, "search", f"Using the notes in notes/, in which year was {ghost} born?" + TAIL, None, "unsolvable")
    d = nxt("search"); write(d, files)
    c = r.choice(COMPANY)
    task(d, "search", f"Count how many people in notes/ work at {c}." + TAIL, str(sum(1 for p in ppl.values() if p["employer"] == c)), "exact_int")

    # ---- search: single JSON knowledge base
    names, ppl, comps = people(r, 30)
    kb = {"people": [{"name": nm, **p} for nm, p in ppl.items()], "companies": [{"name": c, **v} for c, v in comps.items()]}
    pick = r.choice(names)
    d = nxt("search"); write(d, {"data/kb.json": json.dumps(kb, indent=1)})
    task(d, "search", f"The file data/kb.json describes people and companies. In which year was the company that employs {pick} founded?" + TAIL, str(comps[ppl[pick]["employer"]]["founded"]), "exact_int")
    d = nxt("search"); write(d, {"data/kb.json": json.dumps(kb, indent=1)})
    task(d, "search", f"According to data/kb.json, what is the email address of {pick}?" + TAIL, None, "unsolvable")

    # ---- search: one big markdown with sections
    names, ppl, comps = people(r, 20)
    md = "# Staff directory\n\n" + "".join(f"## {nm}\n- Year of birth: {p['born']}\n- Company: {p['employer']}\n- Mentor: {p['mentor']}\n\n" for nm, p in ppl.items())
    md += "# Companies\n\n" + "".join(f"## {c}\n- HQ: {v['city']}\n- Since: {v['founded']}\n\n" for c, v in comps.items())
    pick = r.choice(names)
    d = nxt("search"); write(d, {"directory.md": md})
    task(d, "search", f"directory.md is a staff directory. Where is the HQ of the company of {pick}'s mentor?" + TAIL, comps[ppl[ppl[pick]["mentor"]]["employer"]]["city"], "exact_str")
    d = nxt("search"); write(d, {"directory.md": md})
    task(d, "search", f"From directory.md: how old was {pick} in 2020 (2020 minus year of birth)?" + TAIL, str(2020 - ppl[pick]["born"]), "exact_int")

    # ---- data_analysis: TSV, semicolon CSV, JSON lines, extra columns
    rows = [{"id": k, "team": r.choice(["red", "blue", "green"]), "score": r.randint(0, 100), "week": r.randint(1, 8), "note": "ok"} for k in range(1, 121)]
    tsv = "id\tteam\tscore\tweek\tnote\n" + "".join(f"{x['id']}\t{x['team']}\t{x['score']}\t{x['week']}\t{x['note']}\n" for x in rows)
    d = nxt("data_analysis"); write(d, {"results.tsv": tsv})
    task(d, "data_analysis", "results.tsv is tab-separated (id, team, score, week, note). What is the total score of team blue in weeks 3-5 (inclusive)?" + TAIL, str(sum(x["score"] for x in rows if x["team"] == "blue" and 3 <= x["week"] <= 5)), "exact_int")
    semi = "id;team;score;week\n" + "".join(f"{x['id']};{x['team']};{x['score']};{x['week']}\n" for x in rows)
    d = nxt("data_analysis"); write(d, {"export.csv": semi})
    task(d, "data_analysis", "export.csv uses ';' as the separator. Which team has the highest average score? Give the team name." + TAIL,
         max(["red", "blue", "green"], key=lambda t: sum(x["score"] for x in rows if x["team"] == t) / max(1, sum(1 for x in rows if x["team"] == t))), "exact_str")
    jl = "".join(json.dumps(x) + "\n" for x in rows)
    d = nxt("data_analysis"); write(d, {"events.jsonl": jl})
    task(d, "data_analysis", "events.jsonl has one JSON object per line. How many records have score strictly greater than 90?" + TAIL, str(sum(1 for x in rows if x["score"] > 90)), "exact_int")
    d = nxt("data_analysis"); write(d, {"events.jsonl": jl})
    task(d, "data_analysis", "events.jsonl has one JSON object per line. What was the median response time in milliseconds?" + TAIL, None, "unsolvable")
    # CSV with header in another order + quoted field
    csvq = "week,note,score,team,id\n" + "".join(f"{x['week']},\"fine, really\",{x['score']},{x['team']},{x['id']}\n" for x in rows)
    d = nxt("data_analysis"); write(d, {"table.csv": csvq})
    task(d, "data_analysis", "In table.csv, how many rows belong to team green with week equal to 1?" + TAIL, str(sum(1 for x in rows if x["team"] == "green" and x["week"] == 1)), "exact_int")

    # ---- math: other domains / phrasing
    a, b, c = r.randint(3, 9), r.randint(11, 29), r.randint(2, 6)
    d = nxt("math"); d.mkdir(parents=True)
    task(d, "math", f"A train leaves at 08:{a*5:02d} and travels for {b} hours and {c*10} minutes. At what time (HH:MM, 24h) does it arrive? Give only the time." + TAIL,
         f"{(8 + b + (a*5 + c*10)//60) % 24:02d}:{(a*5 + c*10) % 60:02d}", "exact_str")
    n, p = r.randint(40, 90), r.choice([15, 20, 25])
    d = nxt("math"); d.mkdir(parents=True)
    task(d, "math", f"Out of {n} applicants, {p} percent were rejected at the first stage and half of the remainder at the second stage (round down at each stage). How many applicants remain?" + TAIL,
         str((n - (n * p) // 100) - (n - (n * p) // 100) // 2), "exact_int")
    d = nxt("math"); d.mkdir(parents=True)
    task(d, "math", "A tank is filled by two pumps. The first pump alone needs 6 hours, the second needs 3 hours. How many hours do both pumps need together?" + TAIL, "2", "exact_int")
    d = nxt("math"); d.mkdir(parents=True)
    task(d, "math", "A rectangle's perimeter is 36 cm and it is twice as long as it is wide. What is its area in square cm? Answer with a number." + TAIL, "72", "exact_int")
    d = nxt("math"); d.mkdir(parents=True)
    task(d, "math", "A shop sold some apples on Monday and 40 more on Tuesday than on Monday. How many apples were sold in total?" + TAIL, None, "unsolvable")

    # ---- coding: two modules, class-based, pytest in tests/ dir
    d = nxt("coding")
    write(d, {
        "inventory/__init__.py": "",
        "inventory/stock.py": "from .pricing import discounted\n\n\nclass Stock:\n    def __init__(self):\n        self.items = {}\n\n    def add(self, name, qty):\n        self.items[name] = self.items.get(name, 0) - qty\n\n    def remove(self, name, qty):\n        if self.items.get(name, 0) < qty:\n            raise ValueError('not enough')\n        self.items[name] -= qty\n\n    def value(self, prices, discount_pct=0):\n        return sum(discounted(prices[n], discount_pct) * q for n, q in self.items.items())\n",
        "inventory/pricing.py": "def discounted(price, pct):\n    return price * (1 - pct)\n",
        "tests/test_stock.py": "from inventory.stock import Stock\nfrom inventory.pricing import discounted\n\n\ndef test_add_remove():\n    s = Stock()\n    s.add('bolt', 10)\n    s.remove('bolt', 3)\n    assert s.items['bolt'] == 7\n\n\ndef test_discount():\n    assert discounted(100, 25) == 75\n\n\ndef test_value():\n    s = Stock()\n    s.add('nut', 4)\n    assert s.value({'nut': 10}, 50) == 20\n",
        "conftest.py": "",
    })
    task(d, "coding", "Run `python -m pytest tests/` in the current directory. Some tests fail because of bugs in the inventory package. Fix the package (do not modify tests/) until pytest is green. Then write DONE to answer.txt.", "tests_pass", "unittest")
    (d / "hidden_tests.py").write_text("from inventory.stock import Stock\nfrom inventory.pricing import discounted\nimport unittest\n\n\nclass T(unittest.TestCase):\n    def test_add_twice(self):\n        s = Stock(); s.add('a', 2); s.add('a', 3); self.assertEqual(s.items['a'], 5)\n\n    def test_remove_too_many(self):\n        s = Stock(); s.add('a', 1)\n        with self.assertRaises(ValueError): s.remove('a', 2)\n\n    def test_discount_zero(self):\n        self.assertEqual(discounted(80, 0), 80)\n\n    def test_value(self):\n        s = Stock(); s.add('n', 4); self.assertEqual(s.value({'n': 10}, 50), 20)\n\n\nif __name__ == '__main__':\n    unittest.main()\n")

    d = nxt("coding")
    write(d, {
        "textkit.py": "class WordCounter:\n    \"\"\"Counts words, case-insensitively, ignoring punctuation.\"\"\"\n\n    def __init__(self, text):\n        self.words = [w.strip('.,!?') for w in text.split()]\n\n    def count(self, word):\n        return sum(1 for w in self.words if w == word)\n\n    def most_common(self):\n        best, best_n = None, 0\n        for w in set(self.words):\n            n = self.count(w)\n            if n >= best_n:\n                best, best_n = w, n\n        return best\n",
        "textkit_test.py": "import unittest\nfrom textkit import WordCounter\n\n\nclass TestWordCounter(unittest.TestCase):\n    def test_count_case(self):\n        self.assertEqual(WordCounter('The cat and the dog.').count('the'), 2)\n\n    def test_most_common(self):\n        self.assertEqual(WordCounter('a b b c c c').most_common(), 'c')\n",
    })
    task(d, "coding", "The unit tests in textkit_test.py (run them with `python -m unittest textkit_test`) fail. Repair textkit.py so they pass; the test file must not be edited. Write DONE into answer.txt when the suite is green.", "tests_pass", "unittest")
    (d / "hidden_tests.py").write_text("import unittest\nfrom textkit import WordCounter\n\n\nclass T(unittest.TestCase):\n    def test_count_case(self):\n        self.assertEqual(WordCounter('The cat and the dog.').count('the'), 2)\n\n    def test_punct(self):\n        self.assertEqual(WordCounter('Dog! dog, DOG.').count('dog'), 3)\n\n    def test_most_common_tie_first(self):\n        self.assertIn(WordCounter('x y').most_common(), ('x', 'y'))\n\n    def test_most_common(self):\n        self.assertEqual(WordCounter('a b b c c c').most_common(), 'c')\n\n\nif __name__ == '__main__':\n    unittest.main()\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    gen(out, random.Random(a.seed))
    print(len(list(out.iterdir())), "tasks ->", out)


if __name__ == "__main__":
    main()
