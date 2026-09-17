#!/usr/bin/env python3
"""Reduce the suite to its best N tests, reversibly.

The 100-test suite was raw material. A 20-test suite is what you actually run: cheap enough to
calibrate IN FULL against paid models instead of sampling, and short enough that every test
earns its place. Nothing is deleted - the rest move to tests_archive/ and can be restored.

Selection is scored, not vibes, and DIVERSITY IS ENFORCED: at most `--max-per-category` from any
one category, so the survivors cannot all be C++.

    python3 curate.py --keep 20 --apply
    python3 curate.py --keep 20            # dry run, prints the shortlist and why
    python3 curate.py --restore            # put everything back

Scoring favours a test that: proves its result by COMPILING or RUNNING rather than by regex,
checks that something was removed as well as added, seeds a real codebase to read, and is not
trivially short. It penalises weak checks and unbounded turn counts.
"""
import argparse, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS, ARCHIVE = os.path.join(HERE, "tests"), os.path.join(HERE, "tests_archive")
STRONG = {"command_succeeds", "command_output"}


def score(t, obs=None):
    """Higher is better. Every term is a property of the test file, so this is reproducible."""
    checks = t.get("checks") or []
    types = [c.get("type") for c in checks]
    s, why = 0.0, []

    # Diminishing returns: 25 executable checks is not 6x better than 4, and rewarding raw
    # count let one verbose category swamp the shortlist on the first run of this script.
    strong = sum(1 for x in types if x in STRONG)
    s += 4.0 * min(strong, 4) + 0.25 * max(0, strong - 4)
    if strong:
        why.append("%d executable check(s)" % strong)
    else:
        why.append("NO executable check (weak)")

    if "file_absent" in types:
        s += 2.5
        why.append("proves removal/move")
    if "file_count" in types:
        s += 1.0

    seeded = len((t.get("setup") or {}).get("files") or {})
    if seeded >= 4:
        s += 3.0
        why.append("%d seeded files to read" % seeded)
    elif seeded:
        s += 1.0

    n = len(checks)
    s += min(n, 8) * 0.4
    if n < 3:
        s -= 2.0
        why.append("few checks")

    prompt = t.get("prompt", "")
    s += min(len(prompt) / 400.0, 3.0)
    if len(prompt) < 150:
        s -= 2.0
        why.append("thin prompt")

    # A test whose checks are only regex can be satisfied by a convincing stub.
    if strong == 0 and types.count("file_matches") >= len(types) - 1:
        s -= 3.0
        why.append("regex-only, a stub could pass")

    if int(t.get("max_turns", 25)) > 30:
        s -= 1.5
        why.append("turn limit too generous")

    if t.get("difficulty") == "frontier":
        s += 1.5
    if t.get("category") == "smoke":
        s -= 99
        why.append("smoke test, not a suite member")

    if obs:
        stopped, ok = obs.get("stopped_because"), obs.get("ok")
        if stopped == "max_turns":
            # Ambiguous: could be a hard problem, could be an oversized one. Either way it is a
            # worse discriminator than a clean finish, and it costs the most tokens to run.
            s -= 4.0
            why.append("ran out of turns (size, not skill)")
        elif stopped == "harness_error":
            s -= 20.0
            why.append("HARNESS ERROR - broken, not hard")
        elif ok is False:
            s += 5.0
            why.append("finished and FAILED: discriminates")
        elif ok is True:
            s -= 1.5
            why.append("incumbent passes it comfortably")
        if obs.get("wall_s", 0) > 300:
            s -= 1.5
            why.append("slow (%ds)" % obs["wall_s"])
    return round(s, 2), ", ".join(why)


def measured(path):
    """Outcomes from a previous run, used to prefer tests that DISCRIMINATE.

    A test the model fails by exhausting its turn budget tells you the task was big; a test it
    finishes and then fails on the checks tells you it was WRONG. The second is the signal this
    suite exists to capture, so measured runs re-rank toward it. Measured 2026-09-16: 6 of 14
    hardened tests ended in max_turns, which would otherwise have made task size look like skill
    in the paid calibration.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        data = json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return {}
    return {r["id"]: r for r in data.get("tests", [])}


def load(results=None):
    obs = measured(results)
    out = []
    for name in sorted(os.listdir(TESTS)):
        if name.endswith(".json"):
            try:
                t = json.load(open(os.path.join(TESTS, name)))
            except json.JSONDecodeError:
                continue
            sc, why = score(t, obs.get(t.get("id")))
            out.append({"file": name, "id": t.get("id"), "cat": t.get("category", "misc"),
                        "score": sc, "why": why})
    return out


def pick(rows, keep, max_per_cat):
    """Coverage FIRST, then quality.

    A 20-test suite that is 15 build-system tests measures one thing well and the product not
    at all, which is exactly what the unguarded score produced on its first run. So take the
    best test from every category first, then spend the remaining slots on the highest scores.
    """
    live = [r for r in rows if r["score"] >= 0]
    chosen, per = [], {}
    for cat in sorted({r["cat"] for r in live}):
        best = max((r for r in live if r["cat"] == cat), key=lambda x: x["score"])
        chosen.append(best)
        per[cat] = 1
        if len(chosen) >= keep:
            return chosen
    taken = {r["file"] for r in chosen}
    for r in sorted(live, key=lambda x: -x["score"]):
        if r["file"] in taken or per.get(r["cat"], 0) >= max_per_cat:
            continue
        chosen.append(r)
        taken.add(r["file"])
        per[r["cat"]] = per.get(r["cat"], 0) + 1
        if len(chosen) >= keep:
            break
    return sorted(chosen, key=lambda x: -x["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=20)
    ap.add_argument("--max-per-category", type=int, default=3)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--results", help="a run.py --json file; re-ranks toward tests that "
                                      "FINISHED and failed (real discriminators) over tests "
                                      "that merely ran out of turns")
    a = ap.parse_args()

    if a.restore:
        if not os.path.isdir(ARCHIVE):
            print("nothing archived")
            return 0
        n = 0
        for name in os.listdir(ARCHIVE):
            shutil.move(os.path.join(ARCHIVE, name), os.path.join(TESTS, name))
            n += 1
        print("restored %d tests" % n)
        return 0

    rows = load(a.results)
    chosen = pick(rows, a.keep, a.max_per_category)
    ids = {r["file"] for r in chosen}
    per = {}
    for r in chosen:
        per[r["cat"]] = per.get(r["cat"], 0) + 1
    print("KEEPING %d of %d   categories: %s\n" % (len(chosen), len(rows),
                                                   ", ".join("%s=%d" % kv for kv in sorted(per.items()))))
    for r in chosen:
        print("%6.2f  %-40s %-12s %s" % (r["score"], r["id"], r["cat"], r["why"][:60]))
    if not a.apply:
        print("\n(dry run) re-run with --apply to archive the other %d" % (len(rows) - len(chosen)))
        return 0

    os.makedirs(ARCHIVE, exist_ok=True)
    moved = 0
    for r in rows:
        if r["file"] not in ids:
            shutil.move(os.path.join(TESTS, r["file"]), os.path.join(ARCHIVE, r["file"]))
            moved += 1
    print("\narchived %d to tests_archive/ (reversible: python3 curate.py --restore)" % moved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
