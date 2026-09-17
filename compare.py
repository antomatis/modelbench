#!/usr/bin/env python3
"""Run the suite against several models and print one comparison table.

    python3 compare.py --jobs 8 \
      local=qwen3-coder@http://localhost:8000/v1 \
      fable=claude-fable-5-1@http://your-anthropic-compatible-proxy \
      opus=claude-opus-5@http://your-anthropic-compatible-proxy

Each argument is label=model@base. The Anthropic protocol is inferred from the URL; pass
--key for the pool's anchor token. Models run SEQUENTIALLY (each already parallelises its own
tests with --jobs, and two paid models racing each other only muddies the wall-clock numbers),
but every model's results are cached to results/<label>.json so an interrupted comparison
resumes instead of paying twice.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def run_model(label, model, base, jobs, key, force):
    out = os.path.join(RESULTS, "%s.json" % label)
    if os.path.exists(out) and not force:
        print("== %s: reusing %s (--force to re-run) ==" % (label, out))
        return json.load(open(out))
    cmd = [sys.executable, os.path.join(HERE, "run.py"), "--base", base, "--model", model,
           "--jobs", str(jobs), "--json", out]
    if key:
        cmd += ["--key", key]
    print("== %s: %s @ %s ==" % (label, model, base), flush=True)
    subprocess.run(cmd, text=True, timeout=21600)
    try:
        return json.load(open(out))
    except (OSError, json.JSONDecodeError):
        print("  %s produced no results" % label, file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="label=model@base")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--key")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--json", default=os.path.join(RESULTS, "comparison.json"))
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    runs = {}
    for spec in a.specs:
        label, rest = spec.split("=", 1)
        model, base = rest.rsplit("@", 1)
        key = a.key
        r = run_model(label, model, base, a.jobs, key, a.force)
        if r:
            runs[label] = r

    if not runs:
        print("nothing to compare", file=sys.stderr)
        return 2

    labels = list(runs)
    ids = sorted({t["id"] for r in runs.values() for t in r["tests"]})
    by = {l: {t["id"]: t for t in runs[l]["tests"]} for l in labels}

    print("\n%-42s %s" % ("test", "  ".join("%-10s" % l for l in labels)))
    print("-" * (42 + 12 * len(labels)))
    for tid in ids:
        cells = []
        for l in labels:
            t = by[l].get(tid)
            if not t:
                cells.append("%-10s" % "-")
            elif t["ok"]:
                cells.append("%-10s" % "PASS")
            else:
                mark = "turns" if t.get("stopped_because") == "max_turns" else "%d/%d" % (t["passed"], t["total"])
                cells.append("%-10s" % mark)
        print("%-42s %s" % (tid[:42], "  ".join(cells)))

    print("-" * (42 + 12 * len(labels)))
    summary = {}
    for l in labels:
        tests = runs[l]["tests"]
        n = len(tests)
        ok = sum(1 for t in tests if t["ok"])
        summary[l] = {"passed": ok, "n": n, "pct": round(100.0 * ok / n, 1) if n else 0,
                      "max_turns": sum(1 for t in tests if t.get("stopped_because") == "max_turns")}
    print("%-42s %s" % ("PASSED", "  ".join("%-10s" % ("%d/%d" % (summary[l]["passed"], summary[l]["n"])) for l in labels)))
    print("%-42s %s" % ("PERCENT", "  ".join("%-10s" % ("%.0f%%" % summary[l]["pct"]) for l in labels)))
    print("%-42s %s" % ("ran out of turns", "  ".join("%-10s" % summary[l]["max_turns"] for l in labels)))

    # The tests that separate the models are the ones worth keeping.
    disc = [tid for tid in ids
            if len({bool(by[l].get(tid, {}).get("ok")) for l in labels if by[l].get(tid)}) > 1]
    print("\nDISCRIMINATING (models disagree): %d of %d" % (len(disc), len(ids)))
    for tid in disc:
        print("   ", tid)
    dead = [tid for tid in ids if all(by[l].get(tid, {}).get("ok") for l in labels if by[l].get(tid))]
    if dead:
        print("\nEVERY model passes these - they carry no information: %s" % ", ".join(dead))

    json.dump({"summary": summary, "discriminating": disc, "uninformative": dead},
              open(a.json, "w"), indent=2)
    print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
