"""The agent loop under test: give the model tools, run what it asks for, stop when it stops.

Deliberately thin. The point is to measure the MODEL, so this adds no planning, no retries,
no prompt scaffolding beyond a plain system message. Anything clever here would flatter a
weak model and make the benchmark measure the harness instead.
"""
import json
import os
import urllib.error
import urllib.request

from . import sandbox

MAX_TRUNCATION_NUDGES = 6

SYSTEM = (
    "You are a software engineer working in a sandboxed workspace at /w, which is your current "
    "directory. Use the provided tools to actually do the work: create and edit real files, run "
    "real commands, and verify your own result before you finish. Paths are relative to /w. "
    "There is no network. Do not ask the user questions; finish the task. When it is genuinely "
    "done, reply with a short summary and no further tool calls."
)

TOOLS = {
    "Bash": {
        "type": "function",
        "function": {
            "name": "Bash", "description": "Run a shell command in the workspace and return its output.",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]},
        },
    },
    "Write": {
        "type": "function",
        "function": {
            "name": "Write", "description": "Write a file, creating parent directories. Overwrites.",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                           "required": ["path", "content"]},
        },
    },
    "Read": {
        "type": "function",
        "function": {
            "name": "Read", "description": "Read a file and return its contents.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"]},
        },
    },
}


def _shq(s):
    return "'" + str(s).replace("'", "'\"'\"'") + "'"


def _exec(workspace, name, args, network, cmd_timeout):
    if name == "Bash":
        r = sandbox.run(workspace, args.get("command", ""), timeout=cmd_timeout, network=network)
        return r["output"] or "(no output, rc=%d)" % r["rc"]
    if name == "Write":
        path, content = args.get("path", ""), args.get("content", "")
        # Written THROUGH the sandbox so a path like ../../etc cannot escape the workspace.
        script = "mkdir -p \"$(dirname %s)\" && cat > %s <<'MODELBENCH_EOF'\n%s\nMODELBENCH_EOF" % (
            _shq(path), _shq(path), content)
        r = sandbox.run(workspace, script, timeout=cmd_timeout, network=False)
        return "wrote %s" % path if r["rc"] == 0 else "write failed: %s" % r["output"][-300:]
    if name == "Read":
        r = sandbox.run(workspace, "cat %s" % _shq(args.get("path", "")), timeout=30, network=False)
        return r["output"][:12000] if r["rc"] == 0 else "read failed: %s" % r["output"][-200:]
    return "unknown tool %s" % name


def run(endpoint, workspace, test, verbose=False):
    """Drive the model until it stops calling tools or hits max_turns.

    Returns {turns, tool_calls, stopped_because, transcript_tail, error}.
    `stopped_because` matters when reading results: 'max_turns' usually means the model was
    thrashing, which is a different failure from a wrong answer.
    """
    tools = [TOOLS[t] for t in test.get("tools", ["Bash", "Write", "Read"]) if t in TOOLS]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": test["prompt"]}]
    max_turns = int(test.get("max_turns", 25))
    # 2000 was far too small: one Write of a real source file exceeds it.
    # MODELBENCH_MAX_TOKENS lets a rerun raise the ceiling without editing 20 test files:
    # Fable's 3 "truncated" results (2026-09-16) were this cap, not the model.
    max_tokens = int(test.get("max_tokens", os.environ.get("MODELBENCH_MAX_TOKENS", 8000)))
    cmd_timeout = int(test.get("cmd_timeout_s", 120))
    network = bool(test.get("network", False))
    calls = 0
    truncations = 0
    stopped = "finished"

    for turn in range(max_turns):
        try:
            msg = endpoint.chat(messages, tools=tools, max_tokens=max_tokens)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return {"turns": turn, "tool_calls": calls, "stopped_because": "endpoint_error",
                    "error": "%s: %s" % (type(exc).__name__, exc), "transcript_tail": messages[-2:]}

        tcs = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("text") or None,
                         **({"tool_calls": tcs} if tcs else {})})
        if not tcs:
            # ⚠️ A reply cut off at the token limit is NOT a finished turn. Measured 2026-09-16:
            # a model writing a whole module in one Write call blew the 2000-token budget, its
            # tool_use block was truncated away, and the loop recorded a clean "finished" after
            # 4 turns having built nothing. That silently scored the models that write big files
            # in one call far below the ones that write small ones. Let it continue instead.
            if msg.get("stop_reason") == "max_tokens":
                truncations += 1
                if truncations <= MAX_TRUNCATION_NUDGES:
                    messages.append({"role": "user", "content":
                                     "Your previous reply was cut off at the token limit. Continue "
                                     "from where you stopped. Prefer several smaller tool calls."})
                    continue
                stopped = "truncated"
                break
            stopped = "finished"
            break
        for tc in tcs:
            calls += 1
            fn = (tc.get("function") or {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _exec(workspace, fn.get("name", ""), args, network, cmd_timeout)
            if verbose:
                print("   turn %d %s(%s) -> %s" % (
                    turn, fn.get("name"), str(args)[:70], str(result)[:120].replace("\n", " ")))
            messages.append({"role": "tool", "tool_call_id": tc.get("id") or ("call_%d" % calls),
                             "content": str(result)[:8000]})
    else:
        stopped = "max_turns"

    return {"turns": turn + 1, "tool_calls": calls, "stopped_because": stopped,
            "error": None, "transcript_tail": messages[-3:]}
