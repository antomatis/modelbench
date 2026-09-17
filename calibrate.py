#!/usr/bin/env python3
"""Calibrate suite difficulty against a stratified sample, not the whole suite.

The suite targets: a frontier model passes ~50%, the strongest ~80%. Measuring that on all
100 tests against paid models would cost tens of millions of tokens, so this samples N tests
per category (deterministically, by sorted id) and reports the pass rate with a binomial
confidence interval, so you can see whether the sample is actually tight enough to act on.

    python3 calibrate.py --per-category 2 --model claude-opus-5 --api anthropic \
        --base http://your-anthropic-compatible-proxy --key "$(cat ../anchor.token)"
    python3 calibrate.py --per-category 2 --model qwen3-coder \
        --base http://localhost:8000/v1        # free: local GPU

Add --dry-run to see exactly which tests would run, and what that costs, before spending anything.
"""
import argparse, json, math, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sample(per_cat):
    tests = {}
    for name in sorted(os.listdir(os.path.join(HERE, "tests"))):
        if not name.endswith(".json"):
            continue
        t = json.load(open(os.path.join(HERE, "tests", name)))
        if t.get("category") == "smoke":
            continue
        tests.setdefault(t.get("category", "misc"), []).append(t["id"])
    # Deterministic: sorted ids, first N. Same sample every run, so scores are comparable.
    return [tid for cat in sorted(tests) for tid in sorted(tests[cat])[:per_cat]]


def wilson(k, n, z=1.96):
    """Wilson interval - honest about how little a small sample tells you."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api")
    ap.add_argument("--key")
    ap.add_argument("--json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    ids = sample(a.per_category)
    print("sample: %d tests (%d per category, deterministic by sorted id)" % (len(ids), a.per_category))
    if a.dry_run:
        for i in ids:
            print("   ", i)
        print("\nNOT RUN (--dry-run). Each test is a multi-turn agentic loop; against a paid model "
              "budget on the order of 10^5 tokens per test.")
        return 0

    # One run.py invocation with --jobs, not N sequential ones: the GPU serves 64 concurrent
    # requests and the pool is equally happy in parallel, so walking tests one at a time was
    # spending wall clock for nothing.
    out = os.path.join("/tmp", "_cal_%s.json" % a.model.replace("/", "_"))
    cmd = [sys.executable, os.path.join(HERE, "run.py"), "--base", a.base, "--model", a.model,
           "--jobs", str(a.jobs), "--json", out]
    if a.api:
        cmd += ["--api", a.api]
    if a.key:
        cmd += ["--key", a.key]
    keep = set(ids)
    env = dict(os.environ, MODELBENCH_ONLY=",".join(sorted(keep)))
    subprocess.run(cmd, text=True, timeout=14400, env=env)
    try:
        rows = [r for r in json.load(open(out))["tests"] if r["id"] in keep]
    except Exception as exc:
        print("calibration run produced no readable results: %s" % exc, file=sys.stderr)
        return 2

    k = sum(1 for r in rows if r.get("ok"))
    lo, hi = wilson(k, len(rows))
    print("\n%s: %d/%d = %.0f%%   95%% CI [%.0f%%, %.0f%%]" % (a.model, k, len(rows),
                                                              100 * k / len(rows), 100 * lo, 100 * hi))
    print("target band for this suite: frontier ~50%, strongest ~80%")
    if hi - lo > 0.35:
        print("NOTE: that interval is too wide to tune against - raise --per-category before "
              "concluding anything.")
    if a.json:
        json.dump({"model": a.model, "sample": ids, "passed": k, "n": len(rows),
                   "ci95": [lo, hi], "tests": rows}, open(a.json, "w"), indent=2)
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
