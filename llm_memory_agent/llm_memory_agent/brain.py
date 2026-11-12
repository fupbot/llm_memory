"""Brains decide the next action. The agent loop (agent.py) owns the conversation and
the tools; a Brain only maps (system, tools, messages) to the next turn.

- AnthropicBrain: Claude through the Anthropic Messages API with tool use.
- OpenAICompatBrain: any server with an OpenAI-style /chat/completions endpoint and
  tool calling: a local Ollama (or llama.cpp, vLLM, LM Studio), or a cloud provider
  (OpenAI, OpenRouter, Groq, ...).
- KeywordBaselineBrain: no LLM. Drives to location names in the order they appear
  in the command. A floor to compare the LLM against in evaluation.
- ScriptedBrain: replays fixed tool calls, for tests.

make_brain() picks one from a provider name ("anthropic", "ollama", "openai").

The agent loop keeps the conversation in the Anthropic message format (tool_use /
tool_result blocks). OpenAICompatBrain converts it to the OpenAI format on each call
and stores its own replies as Anthropic-style blocks, so the loop is the same for
every provider.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Turn:
    tool_calls: list[ToolCall]
    text: str
    # Assistant content to append to the history unchanged. For Claude this is
    # response.content, so thinking blocks are passed back exactly as received.
    assistant_content: object
    stop_reason: str
    usage: dict = field(default_factory=dict)


class Brain(ABC):
    name: str = "brain"

    @abstractmethod
    def step(self, system: str, tools: list[dict], messages: list[dict]) -> Turn: ...


class AnthropicBrain(Brain):
    """Claude with client-side tools, using a manual tool-use loop (the agent loop
    needs to stop on `finish`, cap the number of steps, and record a trace).

    Defaults: Claude Opus 5.5, adaptive thinking, effort "medium" (set explicitly;
    that's this model's default too), prompt caching (the system prompt and tools
    don't change, so every turn after the first reads them from cache), and
    server-side refusal fallbacks (`fallbacks: "default"`), so a declined request
    is retried on a fallback model in the same call instead of ending the episode.
    """

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        *,
        effort: str = "medium",
        max_tokens: int = 16000,
        use_fallbacks: bool = True,
        client=None,
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.use_fallbacks = use_fallbacks
        self.name = f"anthropic:{model}:{effort}"

    def step(self, system, tools, messages) -> Turn:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
        )
        if self.use_fallbacks:
            response = self.client.beta.messages.create(
                betas=[self.FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            response = self.client.messages.create(**kwargs)

        calls = [ToolCall(b.id, b.name, dict(b.input))
                 for b in response.content if b.type == "tool_use"]
        text = "".join(b.text for b in response.content if b.type == "text")
        u = response.usage
        usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        return Turn(calls, text, response.content, response.stop_reason or "", usage)


class LLMRequestError(RuntimeError):
    """The LLM server could not be reached or returned an error."""


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:500]
        raise LLMRequestError(f"HTTP {exc.code} from {url}: {body}") from None
    except urllib.error.URLError as exc:
        raise LLMRequestError(f"can't reach {url} ({exc.reason}); is the server running?") from None


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# OpenAI finish_reason -> the Anthropic stop_reason values the agent loop checks.
_STOP_REASONS = {"tool_calls": "tool_use", "function_call": "tool_use", "stop": "end_turn",
                 "length": "max_tokens", "content_filter": "refusal"}


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Anthropic-format history (as built by the agent loop) -> OpenAI chat messages."""
    out = [{"role": "system", "content": system}]
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        if msg["role"] == "assistant":
            text = "".join(b["text"] for b in content if b["type"] == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                     for b in content if b["type"] == "tool_use"]
            m = {"role": "assistant", "content": text or None}
            if calls:
                m["tool_calls"] = calls
            out.append(m)
            continue
        texts = []
        for b in content:  # user turn: tool results (and possibly text)
            if b["type"] == "tool_result":
                body = b["content"] if isinstance(b["content"], str) else json.dumps(b["content"])
                if b.get("is_error"):
                    body = f"ERROR: {body}"
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": body})
            elif b["type"] == "text":
                texts.append(b["text"])
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        fn = {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}
        if t.get("strict"):
            fn["strict"] = True
        out.append({"type": "function", "function": fn})
    return out


class OpenAICompatBrain(Brain):
    """An LLM behind an OpenAI-compatible /chat/completions endpoint with tool calling.

    Works with a local Ollama (http://localhost:11434/v1, no key) and with cloud
    providers that speak this API. Uses only the standard library for HTTP.

    Small local models are less reliable at tool calling: models of roughly 14B+
    trained for it (Qwen3, Llama 3.1+, Mistral Small) work best. Reasoning models'
    <think>...</think> text is stripped from the reply.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        provider: str = "openai-compat",
        max_tokens: int = 4096,
        temperature: float | None = None,
        timeout: float = 300.0,
        post: Callable[[str, dict, dict, float], dict] = _post_json,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._post = post
        self.name = f"{provider}:{model}"

    def step(self, system, tools, messages) -> Turn:
        payload = {"model": self.model, "max_tokens": self.max_tokens,
                   "messages": to_openai_messages(system, messages)}
        if tools:
            payload["tools"] = to_openai_tools(tools)
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = self._post(f"{self.base_url}/chat/completions", payload, headers, self.timeout)
        if "error" in data:
            raise LLMRequestError(f"{self.name}: {data['error']}")

        choice = data["choices"][0]
        msg = choice.get("message") or {}
        text = _THINK_RE.sub("", msg.get("content") or "").strip()
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    # Let the tool executor report it back to the model as an error.
                    args = {"arguments_were_not_valid_json": args}
            if not isinstance(args, dict):
                args = {"arguments_were_not_an_object": args}
            calls.append(ToolCall(tc.get("id") or f"call_{len(messages)}_{i}",
                                  fn.get("name", ""), args))

        content = ([{"type": "text", "text": text}] if text else []) + [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]
        finish = choice.get("finish_reason") or ""
        stop = "tool_use" if calls else _STOP_REASONS.get(finish, finish)
        u = data.get("usage") or {}
        usage = {"input_tokens": u.get("prompt_tokens", 0) or 0,
                 "output_tokens": u.get("completion_tokens", 0) or 0}
        return Turn(calls, text, content, stop, usage)


# provider -> (default model, default base URL, env var holding the API key)
PROVIDERS = {
    "anthropic": ("claude-opus-5-5", None, "ANTHROPIC_API_KEY"),
    "ollama": ("qwen3:8b", "http://localhost:11434/v1", None),
    "openai": (None, "https://api.openai.com/v1", "OPENAI_API_KEY"),
}


def _ollama_base_url() -> str:
    """Ollama's own OLLAMA_HOST convention ("host:port" or a URL), else localhost."""
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if not host:
        return PROVIDERS["ollama"][1]
    if "://" not in host:
        host = f"http://{host}"
    if host.count(":") == 1:  # no port given
        host += ":11434"
    return host.rstrip("/") + "/v1"


def make_brain(
    provider: str = "anthropic",
    model: str | None = None,
    *,
    effort: str = "medium",
    base_url: str | None = None,
    api_key: str | None = None,
    client=None,
) -> Brain:
    """Build a Brain for a provider.

    - anthropic: Claude; key from ANTHROPIC_API_KEY (read by the anthropic SDK).
    - ollama:    local Ollama; base URL from --base-url, else OLLAMA_HOST, else
                 localhost:11434. No key.
    - openai:    any OpenAI-compatible cloud API; key from OPENAI_API_KEY, base URL
                 from --base-url (e.g. https://openrouter.ai/api/v1) or OPENAI_BASE_URL.
                 --model is required.
    API keys are read from the environment rather than taken on the command line,
    so they don't show up in shell history or process listings.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {', '.join(PROVIDERS)}")
    default_model, default_url, key_env = PROVIDERS[provider]
    model = model or default_model
    if not model:
        raise ValueError(f"provider {provider!r} needs a model name (--model)")
    if provider == "anthropic":
        return AnthropicBrain(model, effort=effort, client=client)
    if provider == "ollama":
        return OpenAICompatBrain(model, base_url=base_url or _ollama_base_url(),
                                 api_key=api_key, provider="ollama")
    api_key = api_key or os.environ.get(key_env or "")
    if not api_key:
        raise ValueError(f"provider {provider!r} needs an API key in ${key_env}")
    return OpenAICompatBrain(model, base_url=base_url or os.environ.get("OPENAI_BASE_URL")
                             or default_url, api_key=api_key, provider=provider)


class KeywordBaselineBrain(Brain):
    """Non-LLM baseline: navigate to each known location name mentioned in the
    command, in order of mention, then report success. It can't answer questions,
    search for objects, or recover from failures."""

    name = "baseline:keyword"

    def __init__(self, location_names: list[str]):
        self.location_names = location_names

    def _plan(self, command: str) -> list[str]:
        text = command.lower().replace("_", " ")
        found = []
        for name in self.location_names:
            for m in re.finditer(r"\b" + re.escape(name.replace("_", " ")) + r"\b", text):
                found.append((m.start(), name))
        return [name for _, name in sorted(found)]

    def step(self, system, tools, messages) -> Turn:
        command = messages[0]["content"].rsplit("Operator command:", 1)[-1]
        plan = self._plan(command)
        done = sum(1 for m in messages if m["role"] == "assistant")
        n = len(messages)
        if done < len(plan):
            call = ToolCall(f"call_{n}", "navigate_to",
                            {"location": plan[done], "via": [], "avoid_doors": []})
        else:
            call = ToolCall(f"call_{n}", "finish",
                            {"success": bool(plan), "answer": f"Visited: {', '.join(plan)}"})
        content = [{"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}]
        return Turn([call], "", content, "tool_use")


class ScriptedBrain(Brain):
    """Replays a fixed list of (tool_name, input) calls, one per turn."""

    name = "scripted"

    def __init__(self, calls: list[tuple[str, dict]]):
        self.calls = list(calls)
        self.seen_messages: list[list[dict]] = []

    def step(self, system, tools, messages) -> Turn:
        self.seen_messages.append(list(messages))
        if not self.calls:
            return Turn([], "I'm done.", [{"type": "text", "text": "I'm done."}], "end_turn")
        name, args = self.calls.pop(0)
        call = ToolCall(f"call_{len(messages)}", name, args)
        content = [{"type": "tool_use", "id": call.id, "name": name, "input": args}]
        return Turn([call], "", content, "tool_use")
