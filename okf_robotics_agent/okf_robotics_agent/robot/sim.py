"""SimRobot: a deterministic topological simulator standing in for Nav2 + Gazebo in
evaluation and tests. It runs in microseconds, so an eval of dozens of episodes costs
only LLM calls, and the ground truth for grading is exact.

Behaviour copies what matters from Nav2:
- The planner plans on the static map and doesn't know a door is closed until the
  robot reaches it. Then navigation aborts and the robot stops on the near side.
- A door found closed is remembered for the rest of the *session* (like an obstacle
  marked in a costmap), so a retry in the same session plans around it. Across
  sessions that knowledge is lost unless the agent's long-term memory carries it.
"""
from __future__ import annotations

from ..world import World
from .base import NavResult, Observation, RobotInterface, RobotState


class SimRobot(RobotInterface):
    def __init__(self, world: World):
        self.world = world
        self.location: str = world.start
        self.odometer = 0.0
        self.session_blocked_doors: set[str] = set()
        self.trajectory: list[str] = [world.start]  # every location passed through
        self.nav_calls: list[NavResult] = []

    @property
    def environment_resource(self) -> str:
        return self.world.resource

    def reset_session(self) -> None:
        self.session_blocked_doors.clear()

    def describe_map(self) -> str:
        return self.world.describe_map()

    def state(self) -> RobotState:
        loc = self.world.locations[self.location]
        return RobotState(self.location, loc.x, loc.y, round(self.odometer, 2))

    def look(self) -> Observation:
        objects = sorted(o for o, loc in self.world.objects.items() if loc == self.location)
        return Observation(self.location, objects, self.world.locations[self.location].description)

    def navigate_to(self, location, *, via=None, avoid_doors=None) -> NavResult:
        result = self._navigate(location, list(via or []), set(avoid_doors or []))
        self.nav_calls.append(result)
        return result

    def _navigate(self, target: str, via: list[str], avoid: set[str]) -> NavResult:
        for name in [*via, target]:
            if name not in self.world.locations:
                return NavResult(False, self.location,
                                 f"Unknown location {name!r}. Known locations: "
                                 f"{', '.join(self.world.locations)}.")
        unknown_doors = avoid - self.world.doors
        if unknown_doors:
            return NavResult(False, self.location, f"Unknown door(s) {sorted(unknown_doors)}.")

        excluded = avoid | self.session_blocked_doors
        travelled: list[str] = []
        distance = 0.0
        for waypoint in [*via, target]:
            plan = self.world.shortest_path(self.location, waypoint, excluded_doors=excluded)
            if plan is None:
                return NavResult(
                    False, self.location,
                    f"No route from {self.location} to {waypoint} avoiding doors "
                    f"{sorted(excluded) or '[]'}. Robot stayed at {self.location}.",
                    travelled, round(distance, 2),
                )
            for a, b in zip(plan, plan[1:]):
                door = self.world.door_between(a, b)
                if door and door in self.world.closed_doors:
                    self.session_blocked_doors.add(door)
                    return NavResult(
                        False, self.location,
                        f"Navigation aborted: door {door!r} between {a} and {b} is closed "
                        f"(path blocked). Robot stopped at {self.location}.",
                        travelled, round(distance, 2), blocked_door=door,
                    )
                step = self.world.distance(a, b)
                distance += step
                self.odometer += step
                self.location = b
                travelled.append(b)
                self.trajectory.append(b)
        return NavResult(True, self.location, f"Arrived at {self.location}.",
                         travelled, round(distance, 2))
