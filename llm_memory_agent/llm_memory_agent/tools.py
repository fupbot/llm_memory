"""Tool definitions the LLM sees, and their execution against a RobotInterface and
EpisodicMemory. Tools are coarse (navigate to a named place, look) rather than
velocity commands: the LLM plans and Nav2 (or the simulator) does the motion
control, so model latency never sits inside a control loop."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .memory import EpisodicMemory
from .robot.base import RobotInterface


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


NAVIGATE_TO = {
    "name": "navigate_to",
    "description": (
        "Drive the robot to a named location from the map and wait until it arrives or "
        "fails. The planner uses the static map and does not know which doors are "
        "currently closed: if it reaches a closed door, navigation aborts and the robot "
        "stops before the door. Use `via` to force waypoints in order, and `avoid_doors` "
        "to plan around doors you know or suspect are closed (e.g. from memory)."
    ),
    "input_schema": _schema(
        {
            "location": {"type": "string", "description": "Target location name from the map."},
            "via": {"type": "array", "items": {"type": "string"},
                    "description": "Waypoint location names to pass through, in order. Empty list for none."},
            "avoid_doors": {"type": "array", "items": {"type": "string"},
                            "description": "Door names the route must not cross. Empty list for none."},
        },
        ["location", "via", "avoid_doors"],
    ),
}

LOOK = {
    "name": "look",
    "description": "Report the objects the robot's camera sees at its current location.",
    "input_schema": _schema({}, []),
}

GET_ROBOT_STATE = {
    "name": "get_robot_state",
    "description": "Current location name, map coordinates, and meters travelled this session.",
    "input_schema": _schema({}, []),
}

SEARCH_MEMORY = {
    "name": "search_memory",
    "description": (
        "Semantic search over the robot's long-term memory of past sessions: earlier "
        "commands and their outcomes, objects seen and where, doors found closed, and "
        "operator notes. Each result is timestamped; memories can be out of date, so "
        "check with `look` when it matters."
    ),
    "input_schema": _schema(
        {"query": {"type": "string", "description": "What to look for, in natural language."}},
        ["query"],
    ),
}

REMEMBER = {
    "name": "remember",
    "description": (
        "Save a fact the operator told you that will matter in future sessions (e.g. "
        "a name they use for a place). Things you observe are saved automatically."
    ),
    "input_schema": _schema({"fact": {"type": "string"}}, ["fact"]),
}

FINISH = {
    "name": "finish",
    "description": (
        "End the task. Call exactly once, as your last action. `success` is true only "
        "if the command was actually completed as asked; if it is impossible or you "
        "could not complete it, set false and say why. `answer` is your reply to the "
        "operator (the answer to their question, or a one-line summary)."
    ),
    "input_schema": _schema(
        {"success": {"type": "boolean"}, "answer": {"type": "string"}},
        ["success", "answer"],
    ),
}

ROBOT_TOOLS = [NAVIGATE_TO, LOOK, GET_ROBOT_STATE]
MEMORY_TOOLS = [SEARCH_MEMORY, REMEMBER]


def tool_definitions(memory_enabled: bool) -> list[dict]:
    tools = ROBOT_TOOLS + (MEMORY_TOOLS if memory_enabled else []) + [FINISH]
    return [{**t, "strict": True} for t in tools]


@dataclass
class ToolOutcome:
    content: str
    is_error: bool = False
    finished: bool = False
    success: bool | None = None
    answer: str = ""


@dataclass
class EpisodeTrace:
    """Everything that happened during one command, for memory and evaluation."""

    route: list[str] = field(default_factory=list)
    nav_failures: list[str] = field(default_factory=list)
    blocked_doors: list[tuple[str, str, str]] = field(default_factory=list)
    observations: list[tuple[str, list[str]]] = field(default_factory=list)
    remembered: list[str] = field(default_factory=list)
    distance: float = 0.0


class ToolExecutor:
    def __init__(self, robot: RobotInterface, memory: EpisodicMemory | None):
        self.robot = robot
        self.memory = memory if (memory and memory.enabled) else None
        self.trace = EpisodeTrace()

    def execute(self, name: str, args: dict) -> ToolOutcome:
        try:
            handler = getattr(self, f"_tool_{name}")
        except AttributeError:
            return ToolOutcome(f"Unknown tool {name!r}.", is_error=True)
        try:
            return handler(**args)
        except TypeError as exc:  # wrong/missing arguments
            return ToolOutcome(f"Invalid arguments for {name}: {exc}", is_error=True)

    def _tool_navigate_to(self, location: str, via: list[str] | None = None,
                          avoid_doors: list[str] | None = None) -> ToolOutcome:
        start = self.robot.state().location
        res = self.robot.navigate_to(location, via=via or [], avoid_doors=avoid_doors or [])
        self.trace.route.extend(res.path)
        self.trace.distance += res.distance
        if not res.success:
            self.trace.nav_failures.append(res.message)
            if res.blocked_door:
                here = res.location or start or "?"
                self.trace.blocked_doors.append((res.blocked_door, here, location))
        payload = {"success": res.success, "message": res.message,
                   "location": res.location, "path": res.path, "distance_m": res.distance}
        return ToolOutcome(json.dumps(payload), is_error=False)

    def _tool_look(self) -> ToolOutcome:
        obs = self.robot.look()
        if obs.location:
            self.trace.observations.append((obs.location, obs.objects))
        return ToolOutcome(json.dumps({"location": obs.location, "objects": obs.objects,
                                       "description": obs.description}))

    def _tool_get_robot_state(self) -> ToolOutcome:
        s = self.robot.state()
        return ToolOutcome(json.dumps({"location": s.location, "x": s.x, "y": s.y,
                                       "odometer_m": s.odometer}))

    def _tool_search_memory(self, query: str) -> ToolOutcome:
        if self.memory is None:
            return ToolOutcome("Long-term memory is disabled.", is_error=True)
        hits = self.memory.retrieve(query, k=5)
        if not hits:
            return ToolOutcome("No relevant memories found.")
        return ToolOutcome("\n\n".join(f"({h.path})\n{h.text}" for h in hits))

    def _tool_remember(self, fact: str) -> ToolOutcome:
        if self.memory is None:
            return ToolOutcome("Long-term memory is disabled.", is_error=True)
        self.trace.remembered.append(fact)
        self.memory.record_note(fact)
        self.memory.reindex()
        return ToolOutcome("Saved.")

    def _tool_finish(self, success: bool, answer: str) -> ToolOutcome:
        return ToolOutcome("Task ended.", finished=True, success=bool(success), answer=answer)
