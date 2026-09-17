"""Confined command execution for modelbench.

Every command a model asks for runs through here. The workspace is bound read-write at /w,
the system read-only, /tmp is private, and the network is off unless a test opts in.

This refuses to run at all when bwrap is missing rather than falling back to bare `sh`:
the alternative is executing model-generated shell unconfined on the storage box, which is
exactly the accident this module exists to prevent.
"""
import os
import shutil
import subprocess

BWRAP = shutil.which("bwrap")
RO_PATHS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt", "/usr/local")


class SandboxUnavailable(RuntimeError):
    pass


def _argv(workspace: str, network: bool):
    if not BWRAP:
        raise SandboxUnavailable(
            "bwrap not found; refusing to run model-generated commands unconfined")
    argv = [BWRAP, "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--bind", workspace, "/w", "--chdir", "/w"]
    if not network:
        argv += ["--unshare-net"]
    for p in RO_PATHS:
        if os.path.exists(p):
            argv += ["--ro-bind", p, p]
    # A writable HOME inside the tmpfs keeps tools that insist on one from failing, without
    # exposing the real one.
    argv += ["--setenv", "HOME", "/tmp", "--setenv", "TMPDIR", "/tmp",
             "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin"]
    return argv


def run(workspace: str, command: str, timeout: int = 120, network: bool = False):
    """Run one shell command confined to `workspace`. Never raises on command failure."""
    argv = _argv(workspace, network) + ["/bin/bash", "-lc", command]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        return {"rc": p.returncode, "output": out[:20000], "timed_out": False}
    except subprocess.TimeoutExpired as e:
        partial = ((e.stdout or b"").decode("utf-8", "replace")
                   + (e.stderr or b"").decode("utf-8", "replace")) if e.stdout or e.stderr else ""
        return {"rc": 124, "output": partial[:20000] + "\n[timed out after %ss]" % timeout,
                "timed_out": True}
    except OSError as exc:
        return {"rc": 127, "output": "sandbox failed: %s" % exc, "timed_out": False}


def available() -> bool:
    return BWRAP is not None


def selftest(workspace: str) -> dict:
    """Prove the confinement actually confines, rather than trusting the flags.

    A sandbox nobody drilled is a sandbox nobody knows the shape of, so the runner calls
    this once per session and refuses to continue if any expectation is wrong.
    """
    checks = {}
    r = run(workspace, "touch /w/__probe && echo ok")
    checks["workspace_writable"] = r["rc"] == 0 and "ok" in r["output"]
    r = run(workspace, "touch /usr/__should_fail 2>&1; echo rc=$?")
    checks["system_readonly"] = "rc=0" not in r["output"]
    r = run(workspace, "getent hosts example.com >/dev/null 2>&1; echo rc=$?")
    checks["network_off_by_default"] = "rc=0" not in r["output"]
    r = run(workspace, "ls /base 2>&1; echo rc=$?")
    checks["host_tree_invisible"] = "rc=0" not in r["output"]
    r = run(workspace, "sleep 5", timeout=1)
    checks["timeout_enforced"] = r["timed_out"]
    run(workspace, "rm -f /w/__probe")
    return checks
