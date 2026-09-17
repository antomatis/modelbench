# modelbench — a standard, reusable agentic-coding benchmark

Answers one question: **can this model actually do the work, in a loop, with tools?**

It is not a chat quality probe. Each test drops the model into a **real sandboxed workspace**,
gives it **real tools**, lets it work for up to N turns, and then scores the **workspace it left
behind** — files that exist, code that compiles, tests that pass, folders that moved. The model's
prose is never scored. Only what it built.

Why this exists: a separate floor-probe script is a *floor* — Qwen3-Coder-Next
scores 14/14 on it, so it proves a model is usable but cannot rank two good models. Ranking needs
multi-step work on a real filesystem, which is what this does.

## The 20-test suite

The 100 tests were raw material. What you actually run is the best **20**, chosen by
`curate.py`: cheap enough to calibrate IN FULL against paid models instead of sampling, and
short enough that every test earns its place. Nothing is deleted — the rest sit in
`tests_archive/` and come back with `python3 curate.py --restore`.

```bash
python3 curate.py --keep 20            # dry run: the shortlist, scored, with reasons
python3 curate.py --keep 20 --apply    # archive the rest
```

Selection is **coverage first, then quality**: the best test from every category is taken before
any second test from anywhere. That rule exists because the unguarded score picked 15 build-system
tests on its first run — a suite that measures one thing well and the product not at all. Scoring
rewards executable proof (compiling, running) over regex a stub could satisfy, rewards
`file_absent` checks that prove something really moved, rewards a substantial seeded codebase, and
caps the credit for raw check count so a verbose category cannot swamp the shortlist.

## Difficulty target

Calibrated so a frontier model passes **~50%** and the strongest available passes **~80%**. A test
must be hard because THE PROBLEM is hard — never because the prompt is vague, the expected output
ambiguous, or a check impossible. Every check traces to a sentence in the prompt. Raising the turn
limit is not a difficulty lever; it measures patience.

```bash
python3 calibrate.py --per-category 2 --base http://localhost:8000/v1 --model qwen3-coder
python3 calibrate.py --per-category 2 --base http://your-anthropic-compatible-proxy --api anthropic \
    --model claude-opus-5 --key "$(cat ../anchor.token)" --dry-run
```

`calibrate.py` samples deterministically and reports a Wilson confidence interval, so a small
sample cannot be over-read. `--dry-run` shows what would run before anything is spent.

## Run it

```bash
cd modelbench
python3 run.py --base http://localhost:8000/v1 --model qwen3-coder          # whole suite
python3 run.py --base ... --model ... --filter cpp                               # one category
python3 run.py --base ... --model ... --test cpp-oop-shape-hierarchy --verbose   # one test, see the loop
python3 run.py --base ... --model ... --json results/qwen3-coder.json            # machine-readable
python3 run.py --list                                                            # what exists
```

Two wire protocols are supported and inferred from the URL: OpenAI `/v1/chat/completions` (vLLM
and most local runtimes) and Anthropic `/v1/messages`, including the local rotation pool. So the
same 20 tests measure your GPU and a frontier model on identical work. ⚠️ One confound, stated in
`endpoints.py` too: the pool only serves Claude-Code-shaped requests, so a Claude model receives an
extra identity line in its system prompt that a local model does not. Cross-provider numbers are
indicative; local-vs-local and Claude-vs-Claude are controlled. Results are per-test pass/fail plus a category breakdown.

## The test format (this is the standard — do not invent fields)

One JSON file per test in `tests/`. Everything is optional except `id`, `prompt`, `checks`.

```json
{
  "id": "cpp-oop-shape-hierarchy",
  "category": "cpp",
  "difficulty": "hard",
  "prompt": "What the model is asked to do. Be specific about WHERE files go.",
  "setup": {
    "files": { "src/main.cpp": "…starting content…" },
    "commands": ["git init -q", "mkdir -p include"]
  },
  "tools": ["Bash", "Write", "Read"],
  "max_turns": 25,
  "timeout_s": 600,
  "network": false,
  "checks": [
    { "type": "file_exists",      "path": "include/shape.hpp" },
    { "type": "file_absent",      "path": "src/old.cpp" },
    { "type": "file_matches",     "path": "include/shape.hpp", "regex": "virtual\\s+.*\\s*=\\s*0" },
    { "type": "command_succeeds", "cmd": "g++ -std=c++20 -fsyntax-only -Iinclude src/main.cpp" },
    { "type": "command_output",   "cmd": "./build/app", "contains": "area=12" },
    { "type": "file_count",       "glob": "include/*.hpp", "min": 3 }
  ]
}
```

### Check types

| type | passes when |
|---|---|
| `file_exists` | `path` exists in the workspace |
| `file_absent` | `path` does NOT exist (proves a move/delete really happened) |
| `file_matches` | `path` exists and its text matches `regex` (Python `re.search`, MULTILINE) |
| `file_count` | number of paths matching `glob` is within `min`/`max` |
| `command_succeeds` | `cmd` exits 0 inside the sandbox |
| `command_output` | `cmd` exits 0 AND stdout+stderr `contains` (or matches `regex`) |

Every check runs **inside the same sandbox**, after the model has stopped.

### Writing a good test

1. **Score the artifact, never the prose.** "Explain X" is not a test. "Create X so that Y compiles" is.
2. **Make it fail for the right reason.** Include at least one check that a lazy or partial answer
   fails — a `file_absent` for something that had to move, an executable that must actually run.
3. **State paths explicitly** in the prompt. Ambiguity makes the check wrong, not the model.
4. **No network** unless the test is about network. `"network": false` is the default and keeps runs
   reproducible; npm/pip installs are forbidden by default for the same reason.
5. **Bound it, but do not let the bound do the work.** ⚠️ Measured 2026-09-16: after hardening,
   **17 of 20 tests hit a 30-turn cap and 5 of those still passed every check** — the cap was
   truncating work in progress, not catching thrashing. A tight cap does not make a suite harder,
   it penalises whichever model is less turn-efficient and confounds capability with style. The
   budget is now 60 and the question a test answers is "can it do this"; **turns used is reported
   separately as an efficiency number**. Only treat `max_turns` as a real failure when a model
   burns the budget while making no progress.
6. **Prefer compile/run over grep.** `command_succeeds` on a compiler is far stronger evidence than
   a regex that a stub could satisfy.

## The sandbox

Each test gets a fresh directory under `$TMPDIR`, and every command the model runs is executed with
**bubblewrap**: the workspace bound read-write as `/w`, system paths read-only, a private `/tmp`,
`--die-with-parent`, and **no network** unless the test opts in. The model cannot touch anything on
this box outside its own workspace. Commands are additionally wall-clock bounded.

If `bwrap` is ever missing the runner refuses to execute rather than silently downgrading to
unsandboxed `sh` — running model-generated shell unconfined on the storage box is not a thing that
should happen by accident.

## The `intent` category — comprehension, not coding skill

Every other category asks "can it code". This one asks **whether it understood what was actually
wanted**: the stated constraint, the implicit requirement, the scope it was given, and the scope it
was NOT given. It is the failure mode of weaker models — they pattern-match the literal words,
over-build, ignore a plainly stated preference, or "helpfully" change things nobody asked them to
touch.

It is still scored from artifacts, never prose:

| the model... | proven by |
|---|---|
| honoured a stated constraint | `command_succeeds` / `file_matches` |
| did NOT over-build | `file_count` with a `max`, `file_absent` on what was not asked for |
| broke nothing unrelated | the seeded test suite still passes, untouched files byte-identical |
| kept a public API intact | exact signature match + callers still compile |
| did the prioritised thing | the important artifact exists, the deprioritised one does not |

The fairness line is absolute here: the intent must be unambiguous to a careful reader. Testing
whether a model READ and RESPECTED what was written is fair. Expecting it to intuit something
unstated is a broken test, not a hard one.

## Interpreting a score

- **Tool-calling failures are disqualifying,** not a lost point. A model that will not emit
  structured calls cannot drive an agentic CLI at any benchmark score.
- Compare **category** scores, not just the total: a model can be strong at algorithms and useless
  at multi-file refactoring, and only one of those matters for a given job.
- A test every candidate passes carries no information. Prune it or make it harder.
