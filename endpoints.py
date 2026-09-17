"""Endpoints modelbench can drive, behind one interface.

`agent.py` only ever sees OpenAI-shaped messages and tool calls. Anything else is translated
here, so adding a provider never touches the agent loop or the tests.

  OpenAIEndpoint     - any /v1/chat/completions server (vLLM, and most local runtimes)
  AnthropicEndpoint  - /v1/messages, including the local rotation pool at your-anthropic-compatible-proxy

⚠️ CALIBRATION CONFOUND, stated once so nobody forgets it: the rotation pool only serves a
Claude-Code-shaped request (an `anthropic-beta: oauth-2025-04-20` header and a system block
beginning with the Claude Code identity line); a bare probe comes back as an unexplained 429.
So a Claude model under test receives that identity line PLUS the benchmark's own system
prompt, while a local model receives only the benchmark's. The prompts are therefore not
byte-identical across providers. Treat cross-provider scores as indicative, and compare
local-vs-local or Claude-vs-Claude when you need a controlled number.
"""
import json
import urllib.request

CC_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


class OpenAIEndpoint:
    kind = "openai"

    def __init__(self, base, model, timeout=900, key=None):
        self.base, self.model, self.timeout, self.key = base.rstrip("/"), model, timeout, key

    def chat(self, messages, tools=None, max_tokens=2000):
        body = {"model": self.model, "messages": messages,
                "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = urllib.request.Request(self.base + "/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        d = json.load(urllib.request.urlopen(req, timeout=self.timeout))
        m = d["choices"][0]["message"]
        return {"text": m.get("content") or "", "tool_calls": m.get("tool_calls") or [],
                "stop_reason": (d["choices"][0].get("finish_reason") or "").replace("length", "max_tokens")}


class AnthropicEndpoint:
    """Speaks /v1/messages, presenting the same OpenAI-shaped result to the agent loop."""
    kind = "anthropic"

    def __init__(self, base, model, timeout=900, key=None):
        self.base, self.model, self.timeout = base.rstrip("/"), model, timeout
        self.key = key

    @staticmethod
    def _tools(tools):
        out = []
        for t in tools or []:
            f = t.get("function", t)
            out.append({"name": f["name"], "description": f.get("description", ""),
                        "input_schema": f.get("parameters") or {"type": "object", "properties": {}}})
        return out

    @staticmethod
    def _messages(messages):
        """OpenAI roles -> Anthropic blocks. Tool results become user tool_result blocks."""
        system, out = [], []
        for m in messages:
            role = m.get("role")
            if role == "system":
                system.append(m.get("content") or "")
                continue
            if role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id"),
                         "content": str(m.get("content") or "")[:8000]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)      # consecutive results share one turn
                else:
                    out.append({"role": "user", "content": [block]})
                continue
            if role == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append({"type": "tool_use", "id": tc.get("id") or "call_1",
                                   "name": fn.get("name"), "input": args})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "."}]})
                continue
            out.append({"role": "user", "content": m.get("content") or ""})
        return system, out

    def chat(self, messages, tools=None, max_tokens=2000):
        system, msgs = self._messages(messages)
        # The identity line must come first or the pool refuses the request (see module docstring).
        sys_blocks = [{"type": "text", "text": CC_IDENTITY}] + \
                     [{"type": "text", "text": s} for s in system if s]
        # NO temperature: the current Claude models reject it outright ("`temperature` is
        # deprecated for this model", 400, measured 2026-09-16). So the local side runs greedy
        # at temperature 0 and the Claude side runs at its own default. That asymmetry cannot
        # be removed from this end - record it and read single runs with the variance in mind
        # rather than pretending the two are identically sampled.
        body = {"model": self.model, "max_tokens": max_tokens,
                "system": sys_blocks, "messages": msgs}
        if tools:
            body["tools"] = self._tools(tools)
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01",
                   "anthropic-beta": "oauth-2025-04-20"}
        if self.key:
            headers["authorization"] = "Bearer " + self.key
        req = urllib.request.Request(self.base + "/v1/messages",
                                     data=json.dumps(body).encode(), headers=headers)
        d = json.load(urllib.request.urlopen(req, timeout=self.timeout))
        text, calls = "", []
        for blk in d.get("content") or []:
            if blk.get("type") == "text":
                text += blk.get("text") or ""
            elif blk.get("type") == "tool_use":
                calls.append({"id": blk.get("id"), "type": "function",
                              "function": {"name": blk.get("name"),
                                           "arguments": json.dumps(blk.get("input") or {})}})
        return {"text": text, "tool_calls": calls, "stop_reason": d.get("stop_reason")}


def build(base, model, api=None, key=None, timeout=900):
    """Pick the adapter. `api` forces it; otherwise infer from the URL."""
    if api == "anthropic" or (api is None and base.rstrip("/").endswith("/v1/messages")):
        return AnthropicEndpoint(base.rstrip("/").removesuffix("/v1/messages"), model, timeout, key)
    return OpenAIEndpoint(base, model, timeout, key)
