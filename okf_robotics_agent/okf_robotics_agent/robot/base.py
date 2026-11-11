"""RobotInterface: the only surface the LLM agent uses to act on the robot. Two
implementations: SimRobot (built-in topological simulator, used for evaluation and
tests) and Nav2Robot (ROS 2 Nav2 action clients, used against Gazebo or real
hardware). The agent and its tools can't tell which one they're driving."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class NavResult:
    success: bool
    location: str | None  # named location the robot is at afterwards (None if between)
    message: str
    path: list[str] = field(default_factory=list)
    distance: float = 0.0
    blocked_door: str | None = None


@dataclass
class Observation:
    location: str | None
    objects: list[str]
    description: str = ""


@dataclass
class RobotState:
    location: str | None
    x: float
    y: float
    odometer: float  # meters travelled this session


class RobotInterface(ABC):
    @abstractmethod
    def navigate_to(
        self, location: str, *, via: list[str] | None = None, avoid_doors: list[str] | None = None
    ) -> NavResult:
        """Drive to a named location, optionally through `via` waypoints in order and
        never through any door in `avoid_doors`. Blocks until arrival or failure."""

    @abstractmethod
    def look(self) -> Observation:
        """What the robot perceives at its current location."""

    @abstractmethod
    def state(self) -> RobotState: ...

    @abstractmethod
    def describe_map(self) -> str:
        """Static map summary (locations, connections, doors) for the agent prompt."""

    @property
    @abstractmethod
    def environment_resource(self) -> str:
        """OKF `resource` URI identifying this environment, e.g. "sim://house"."""

    def reset_session(self) -> None:
        """Forget session-local state (e.g. doors seen blocked this session) at the
        start of a new session. No-op by default."""
