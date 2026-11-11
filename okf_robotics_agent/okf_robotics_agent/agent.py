"""RobotAgent: turns one natural-language operator command into robot actions.

Each command runs one episode:
1. Retrieve: the command is used as a query against long-term memory, and the top-k
   memories go into the first user message. The model can also call
   `search_memory` for more.
2. Act: tool-use loop (navigate_to / look / get_robot_state / ...) until the model
   calls `finish` or hits the step limit.
3. Write back: the episode (route, failures, observations, outcome) is written to
   the OKF bundle and re-indexed, so the next session can retrieve it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .brain import Brain
from .memory import EpisodeRecord, EpisodicMemory
from .robot.base import RobotInterface
from .tools import EpisodeTrace, ToolExecutor, tool_definitions

SYSTEM_PROMPT = """\
You control a mobile robot (a differential-drive base with a camera, no arm) through \
tools. An operator gives you one command in natural language; carry it out, then call \
`finish` with whether you actually completed it and your reply to the operator.

How the robot works:
- `navigate_to` drives to named map locations. Motion control and obstacle avoidance \
are handled for you. The planner doesn't know which doors are closed until the robot \
reaches one; then navigation aborts and you can try another route.
- `look` tells you which objects are at the current location. That is the only way to \
confirm where an object is right now.
{memory_section}
Be honest in `finish`: if a place doesn't exist on the map, a route is blocked with \
no alternative, or you couldn't find what was asked, report success=false and say why. \
Don't claim anything you didn't observe. Keep the robot's travel reasonable: don't \
visit places the command doesn't need.

Map of {environment}:
{map}
"""

MEMORY_SECTION = """\
- You have long-term memory of earlier sessions in this environment, through \
`search_memory` and the memories included with each command. Memories are \
timestamped and can be out of date: objects get moved; doors may have been reopened. \
Use them to plan (e.g. pass a door that was closed last time in `avoid_doors`, or go \
first to where an object was last seen), then confirm with `look`. Use `remember` for \
facts the operator tells you that will matter later.
"""

NO_MEMORY_SECTION = """\
- You have no memory of earlier sessions.
"""


@dataclass
class EpisodeResult:
    command: str
    success_claimed: bool | None
    answer: str
    steps: int
    final_location: str | None
    tool_calls: list[dict] = field(default_factory=list)
    trace: EpisodeTrace = field(default_factory=EpisodeTrace)
    memories_injected: list[str] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class RobotAgent:
    def __init__(
        self,
        robot: RobotInterface,
        brain: Brain,
        memory: EpisodicMemory | None = None,
        *,
        max_steps: int = 20,
        retrieve_k: int = 5,
    ):
        self.robot = robot
        self.brain = brain
        self.memory = memory
        self.max_steps = max_steps
        self.retrieve_k = retrieve_k

    @property
    def memory_enabled(self) -> bool:
        return self.memory is not None and self.memory.enabled

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            memory_section=MEMORY_SECTION if self.memory_enabled else NO_MEMORY_SECTION,
            environment=self.robot.environment_resource,
            map=self.robot.describe_map(),
        )

    def _first_message(self, command: str, memories: list) -> str:
        parts = []
        if self.memory_enabled:
            if memories:
                block = "\n\n".join(f"<memory source=\"{m.path}\">\n{m.text}\n</memory>"
                                    for m in memories)
                parts.append(f"Possibly relevant memories from earlier sessions:\n{block}")
            else:
                parts.append("No memories from earlier sessions matched this command.")
        state = self.robot.state()
        parts.append(f"The robot is at {state.location}.")
        parts.append(f"Operator command: {command}")
        return "\n\n".join(parts)

    def run(self, command: str) -> EpisodeResult:
        memories = self.memory.retrieve(command, self.retrieve_k) if self.memory_enabled else []
        executor = ToolExecutor(self.robot, self.memory)
        system = self.system_prompt()
        tools = tool_definitions(self.memory_enabled)
        messages: list[dict] = [{"role": "user", "content": self._first_message(command, memories)}]
        result = EpisodeResult(command, None, "", 0, None,
                               memories_injected=[m.path for m in memories])
        usage: dict[str, int] = {}
        nudged = False

        try:
            while result.steps < self.max_steps:
                turn = self.brain.step(system, tools, messages)
                result.steps += 1
                result.stop_reason = turn.stop_reason
                for key, val in turn.usage.items():
                    usage[key] = usage.get(key, 0) + val
                messages.append({"role": "assistant", "content": turn.assistant_content})

                if turn.stop_reason == "refusal":
                    result.answer = "The model declined this request."
                    break
                if not turn.tool_calls:
                    if nudged or turn.stop_reason == "max_tokens":
                        result.answer = turn.text
                        break
                    nudged = True
                    messages.append({"role": "user", "content":
                                     "Call `finish` to end the task, or keep working if you aren't done."})
                    continue

                tool_results, finished = [], None
                for call in turn.tool_calls:
                    outcome = executor.execute(call.name, call.input)
                    result.tool_calls.append({"name": call.name, "input": call.input,
                                              "result": outcome.content, "is_error": outcome.is_error})
                    tool_results.append({"type": "tool_result", "tool_use_id": call.id,
                                         "content": outcome.content, "is_error": outcome.is_error})
                    if outcome.finished:
                        finished = outcome
                if finished:
                    result.success_claimed = finished.success
                    result.answer = finished.answer
                    break
                messages.append({"role": "user", "content": tool_results})
            else:
                result.answer = result.answer or f"Stopped after {self.max_steps} steps."
        except Exception as exc:  # API errors etc.: record, still write memory below
            result.error = f"{type(exc).__name__}: {exc}"

        result.trace = executor.trace
        result.usage = usage
        result.final_location = self.robot.state().location
        if self.memory_enabled:
            self.memory.record_episode(EpisodeRecord(
                command=command,
                success_claimed=result.success_claimed,
                answer=result.answer,
                route=executor.trace.route,
                nav_failures=executor.trace.nav_failures,
                blocked_doors=executor.trace.blocked_doors,
                observations=executor.trace.observations,
            ))
        return result


def format_result(result: EpisodeResult) -> str:
    """Human-readable one-episode summary for the CLI and the ROS node."""
    lines = [f"> {result.command}"]
    for call in result.tool_calls:
        args = json.dumps(call["input"]) if call["input"] else ""
        lines.append(f"  {call['name']}({args}) -> {call['result'][:160]}")
    status = {True: "SUCCESS", False: "FAILED", None: "NO REPORT"}[result.success_claimed]
    lines.append(f"[{status}] {result.answer}")
    if result.error:
        lines.append(f"[error] {result.error}")
    return "\n".join(lines)
