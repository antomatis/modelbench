# modelbench — run it against YOUR local model in 3 commands


A 20-test agentic coding benchmark. Each test drops the model into a fresh sandbox with real
tools (Bash / Write / Read), lets it work for up to 60 turns, then scores THE FILES IT LEFT BEHIND:
does it compile, does it run, does the output match, did the old file really move. The model's
prose is never scored. A model that explains beautifully and builds nothing scores zero.

## Requirements (Linux)
- `python3` (3.9+, stdlib only — nothing to pip install)
- `bwrap` (bubblewrap) — the sandbox. `apt install bubblewrap` / `dnf install bubblewrap`.
  The runner REFUSES to run without it rather than executing model-generated shell unconfined.
- The tests use: `g++`, `cmake`, `make`, `node`, `python3`, `git`, `bash`. Install what you have;
  a missing toolchain fails only the tests that need it.
- No network is needed or allowed inside the sandbox.

## Run
```bash
git clone https://github.com/antomatis/modelbench.git && cd modelbench
python3 run.py --list                                                      # what is in the suite
python3 run.py --base http://localhost:8000/v1 --model my-model --jobs 4    # any OpenAI-compatible server
python3 run.py --base ... --model ... --test cpp-break-include-cycle --verbose   # watch one test's tool calls
python3 run.py --base ... --model ... --json results/my-model.json         # machine-readable
```
`--base` is any OpenAI `/v1/chat/completions` endpoint: vLLM, llama.cpp server, LM Studio, Ollama's
OpenAI mode, etc. Your model MUST support function/tool calling on that endpoint — a model that
cannot emit structured tool calls scores 0 here by design (see README "Interpreting a score").
`--jobs` runs tests concurrently; vLLM batches them, so 4-8 is fine on one GPU.

## Compare two models
```bash
python3 compare.py --jobs 4 mine=my-model@http://localhost:8000/v1 other=other-model@http://otherhost:8000/v1
```
Prints one table and names the tests the models actually DISAGREE on — those are the ones that
carry information.

## Reference scores on this exact suite (2026-09-17, same harness)
| model | pass |
|---|---|
| claude-opus-5 | 20/20 |
| claude-fable-5-1 | 19/20 |
| Qwen/Qwen3.8-27B-FP8 (vLLM, 1x 96 GB) | 19/20 |
| Qwen/Qwen3-Coder-Next-FP8 | 14/20 |
| meta-models/Muse-Glimmer-30B | 13/20 |

Two traps we hit so you don't: a reply-token ceiling that is too low truncates a model that writes
whole files in one tool call and silently scores it as "finished" (the runner now continues a
truncated reply; keep `MODELBENCH_MAX_TOKENS` at 8000+); and a tight turn cap measures patience,
not skill (the suite uses 60 and reports turns separately).

Format for writing your own tests: `README.md`. Everything is one JSON file per test.
