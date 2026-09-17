#!/usr/bin/env python3
"""modelbench CLI — run the agentic coding suite against any OpenAI-compatible endpoint.

    python3 run.py --base http://localhost:8000/v1 --model qwen3-coder
    python3 run.py --base ... --model ... --filter cpp --verbose
    python3 run.py --list

See README.md for the test format. The suite scores the workspace a model leaves behind,
so a model that talks well and builds nothing scores zero, which is the intent.
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modelbench import agent, checks, endpoints, sandbox  # noqa: E402

TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")


def load_tests(filt=None, one=None):
    """MODELBENCH_ONLY restricts to a comma-separated id list, so a caller can select a
    subset and still get one parallel run rather than one subprocess per test."""
    only = {x for x in (os.environ.get("MODELBENCH_ONLY") or "").split(",") if x}
    out = []
    for name in sorted(os.listdir(TESTS_DIR)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(TESTS_DIR, name)) as fh:
            try:
                t = json.load(fh)
            except json.JSONDecodeError as exc:
                print("SKIP %s: invalid JSON (%s)" % (name, exc), file=sys.stderr)
                continue
        if one and t.get("id") != one:
            continue
        if filt and filt not in (t.get("category", "") + " " + t.get("id", "")):
            continue
        if only and t.get("id") not in only:
            continue
        out.append(t)
    return out


def prepare(test):
    ws = tempfile.mkdtemp(prefix="modelbench-%s-" % test["id"][:24])
    setup = test.get("setup") or {}
    for path, content in (setup.get("files") or {}).items():
        full = os.path.join(ws, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as fh:
            fh.write(content)
    for cmd in (setup.get("commands") or []):
        sandbox.run(ws, cmd, timeout=120, network=bool(test.get("network")))
    return ws


def run_one(ep, test, verbose=False, keep=False):
    ws = prepare(test)
    t0 = time.monotonic()
    try:
        loop = agent.run(ep, ws, test, verbose=verbose)
        results = checks.run_all(ws, test.get("checks", []), bool(test.get("network")))
    finally:
        if keep:
            print("   workspace kept at %s" % ws)
        else:
            shutil.rmtree(ws, ignore_errors=True)
    passed = sum(1 for _, ok, _ in results if ok)
    return {"id": test["id"], "category": test.get("category", "misc"),
            "difficulty": test.get("difficulty", "hard"),
            "passed": passed, "total": len(results),
            "ok": len(results) > 0 and passed == len(results),
            "wall_s": round(time.monotonic() - t0, 1),
            "turns": loop["turns"], "tool_calls": loop["tool_calls"],
            "stopped_because": loop["stopped_because"], "error": loop.get("error"),
            "checks": [{"type": t, "pass": ok, "detail": d} for t, ok, d in results]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--model")
    ap.add_argument("--filter", help="substring of category or id")
    ap.add_argument("--test", help="exact test id")
    ap.add_argument("--json", help="write full results here")
    ap.add_argument("--verbose", action="store_true", help="print every tool call")
    ap.add_argument("--keep", action="store_true", help="keep workspaces for inspection")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--jobs", type=int, default=8,
                    help="tests to run concurrently. The GPU serves 64 concurrent requests and "
                         "vLLM batches them, so serial runs waste most of the box. 1 to debug.")
    ap.add_argument("--api", choices=["openai", "anthropic"],
                    help="force the wire protocol; inferred from --base when omitted")
    ap.add_argument("--key", help="bearer token; for the rotation pool use the pool anchor token")
    a = ap.parse_args()

    tests = load_tests(a.filter, a.test)
    if a.list:
        by = {}
        for t in tests:
            by.setdefault(t.get("category", "misc"), []).append(t["id"])
        for cat in sorted(by):
            print("%-14s %d" % (cat, len(by[cat])))
            for tid in sorted(by[cat]):
                print("    ", tid)
        print("TOTAL %d tests" % len(tests))
        return 0
    if not a.base or not a.model:
        ap.error("--base and --model are required unless --list")
    if not tests:
        print("no tests matched", file=sys.stderr)
        return 2

    if not sandbox.available():
        print("FATAL: bwrap missing; refusing to run model-generated commands unconfined",
              file=sys.stderr)
        return 3
    probe_ws = tempfile.mkdtemp(prefix="modelbench-selftest-")
    st = sandbox.selftest(probe_ws)
    shutil.rmtree(probe_ws, ignore_errors=True)
    if not all(st.values()):
        print("FATAL: sandbox self-test failed: %s" % st, file=sys.stderr)
        return 3
    print("sandbox ok (%s)" % ", ".join(sorted(st)))

    ep0 = endpoints.build(a.base, a.model, a.api, a.key)
    print("== %s @ %s [%s] == %d tests" % (a.model, a.base, ep0.kind, len(tests)))
    # Each test owns its own sandbox directory and its own endpoint object, so tests are
    # independent and safe to run concurrently. --verbose forces serial, because interleaved
    # tool-call traces from 8 tests are unreadable.
    jobs = 1 if a.verbose else max(1, a.jobs)
    rows = []
    t_start = time.monotonic()

    def work(t):
        return run_one(endpoints.build(a.base, a.model, a.api, a.key), t, a.verbose, a.keep)

    def report(i, r):
        flag = "PASS" if r["ok"] else "FAIL"
        note = "" if r["stopped_because"] == "finished" else " [%s]" % r["stopped_because"]
        print("%3d/%d %-34s %s %d/%d  turns=%d calls=%d %.0fs%s" % (
            i, len(tests), r["id"], flag, r["passed"], r["total"], r["turns"],
            r["tool_calls"], r["wall_s"], note), flush=True)
        if not r["ok"] and a.verbose:
            for c in r["checks"]:
                if not c["pass"]:
                    print("        %s: %s" % (c["type"], c["detail"][:160]))

    if jobs == 1:
        for i, t in enumerate(tests, 1):
            r = work(t)
            rows.append(r)
            report(i, r)
    else:
        print("running %d tests, %d at a time" % (len(tests), jobs), flush=True)
        with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(work, t): t for t in tests}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                try:
                    r = fut.result()
                except Exception as exc:
                    t = futs[fut]
                    r = {"id": t["id"], "category": t.get("category", "misc"), "passed": 0,
                         "total": len(t.get("checks", [])), "ok": False, "wall_s": 0, "turns": 0,
                         "tool_calls": 0, "stopped_because": "harness_error",
                         "error": "%s: %s" % (type(exc).__name__, exc), "checks": []}
                rows.append(r)
                report(i, r)
    wall = time.monotonic() - t_start
    serial = sum(r["wall_s"] for r in rows)
    print("\nwall %.0fs (sum of tests %.0fs, %.1fx from --jobs %d)" % (
        wall, serial, serial / wall if wall else 1, jobs))

    total_ok = sum(1 for r in rows if r["ok"])
    print("\nSUITE %d/%d tests fully passed" % (total_ok, len(rows)))
    by = {}
    for r in rows:
        d = by.setdefault(r["category"], [0, 0])
        d[1] += 1
        d[0] += 1 if r["ok"] else 0
    for cat in sorted(by):
        print("  %-14s %d/%d" % (cat, by[cat][0], by[cat][1]))
    stuck = [r["id"] for r in rows if r["stopped_because"] == "max_turns"]
    if stuck:
        print("  hit max_turns (thrashing, not a wrong answer): %s" % ", ".join(stuck[:8]))
    if any(r["stopped_because"] == "endpoint_error" for r in rows):
        print("  WARNING: endpoint errors occurred; the score is not comparable")

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump({"model": a.model, "base": a.base, "suite_passed": total_ok,
                   "suite_total": len(rows), "by_category": by, "tests": rows},
                  open(a.json, "w"), indent=2)
        print("wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
