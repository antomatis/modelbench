"""Verdicts for modelbench: score the workspace the model left behind, never its prose.

Each check returns (passed, detail). Detail is what a human reads when a test fails, so it
says what was actually found, not just "failed".
"""
import glob as _glob
import os
import re

from . import sandbox


def _abs(workspace, path):
    """Resolve inside the workspace and refuse to escape it."""
    full = os.path.realpath(os.path.join(workspace, path))
    root = os.path.realpath(workspace)
    if full != root and not full.startswith(root + os.sep):
        raise ValueError("check path escapes the workspace: %r" % path)
    return full


def _read(workspace, path):
    with open(_abs(workspace, path), "r", errors="replace") as fh:
        return fh.read()


def file_exists(workspace, c, net):
    ok = os.path.exists(_abs(workspace, c["path"]))
    return ok, "%s %s" % (c["path"], "exists" if ok else "MISSING")


def file_absent(workspace, c, net):
    ok = not os.path.exists(_abs(workspace, c["path"]))
    return ok, "%s %s" % (c["path"], "absent" if ok else "STILL PRESENT")


def file_matches(workspace, c, net):
    try:
        body = _read(workspace, c["path"])
    except OSError:
        return False, "%s missing" % c["path"]
    ok = re.search(c["regex"], body, re.MULTILINE | re.DOTALL) is not None
    return ok, "%s %s /%s/" % (c["path"], "matches" if ok else "DOES NOT MATCH", c["regex"][:60])


def file_count(workspace, c, net):
    hits = _glob.glob(os.path.join(workspace, c["glob"]), recursive=True)
    n = len(hits)
    lo, hi = c.get("min", 0), c.get("max")
    ok = n >= lo and (hi is None or n <= hi)
    return ok, "%s matched %d (min=%s max=%s)" % (c["glob"], n, lo, hi)


def command_succeeds(workspace, c, net):
    r = sandbox.run(workspace, c["cmd"], timeout=c.get("timeout_s", 120), network=net)
    return r["rc"] == 0, "`%s` rc=%d %s" % (c["cmd"][:60], r["rc"], r["output"][-200:].strip())


def command_output(workspace, c, net):
    r = sandbox.run(workspace, c["cmd"], timeout=c.get("timeout_s", 120), network=net)
    if r["rc"] != 0 and not c.get("allow_failure"):
        return False, "`%s` rc=%d %s" % (c["cmd"][:60], r["rc"], r["output"][-200:].strip())
    if "contains" in c:
        ok = c["contains"] in r["output"]
        how = "contains %r" % c["contains"]
    else:
        ok = re.search(c["regex"], r["output"], re.MULTILINE | re.DOTALL) is not None
        how = "matches /%s/" % c["regex"][:50]
    return ok, "`%s` %s%s | out=%s" % (c["cmd"][:50], "" if ok else "NOT ", how, r["output"][-160:].strip())


REGISTRY = {
    "file_exists": file_exists,
    "file_absent": file_absent,
    "file_matches": file_matches,
    "file_count": file_count,
    "command_succeeds": command_succeeds,
    "command_output": command_output,
}


def run_all(workspace, checks, network=False):
    """Every check runs even after one fails: a partial result is far more diagnostic."""
    results = []
    for c in checks:
        fn = REGISTRY.get(c.get("type"))
        if fn is None:
            results.append((c.get("type", "?"), False, "unknown check type"))
            continue
        try:
            ok, detail = fn(workspace, c, network)
        except Exception as exc:                      # a broken check is a failed check, loudly
            ok, detail = False, "check raised %s: %s" % (type(exc).__name__, exc)
        results.append((c["type"], ok, detail))
    return results
