"""Per-type cost breakdown of a traces.jsonl: tasks, spend, input/output tokens, calls per task."""
import collections, json, sys
agg = collections.defaultdict(lambda: dict(n=0, spent=0.0, inp=0, out=0, calls=0))
for f in sys.argv[1:]:
    for l in open(f):
        t = json.loads(l)
        k = ("verify:" if t.get("phase") == "verify" else "") + t["type"]
        a = agg[k]; a["n"] += 1; a["spent"] += t["spent"]
        for x in t["trace"]:
            if "cost" in x:
                a["calls"] += 1; a["inp"] += x.get("in") or 0; a["out"] += x.get("out") or 0
tot = 0
for k, a in sorted(agg.items()):
    n = a["n"]; tot += a["spent"]
    print(f"{k:22} n={n:3} spent/task={a['spent']/n:6.1f}  in/task={a['inp']/n:6.0f} out/task={a['out']/n:5.0f} calls/task={a['calls']/n:.2f}")
print(f"total spent={tot:.0f}")
