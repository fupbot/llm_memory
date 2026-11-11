"""Brains decide the next action. The agent loop (agent.py) owns the conversation and
the tools; a Brain only maps (system, tools, messages) to the next turn.

- AnthropicBrain: Claude through the Anthropic Messages API with tool use.
- KeywordBaselineBrain: no LLM. Drives to location names in the order they appear
  in the command. A floor to compare the LLM against in evaluation.
- ScriptedBrain: replays fixed tool calls, for tests.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
