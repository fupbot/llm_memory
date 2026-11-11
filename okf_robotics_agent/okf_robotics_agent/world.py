"""World: the topological map (locations, connections, doors, objects) shared by the
simulated robot and the agent's prompt. Pure Python, no ROS."""
from __future__ import annotations

import copy
import heapq
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class Location:
    name: str
    x: float
    y: float
    description: str = ""


@dataclass
class Connection:
    a: str
    b: str
    door: str | None = None


@dataclass
class World:
    name: str
    resource: str
    start: str
    locations: dict[str, Location]
    connections: list[Connection]
    objects: dict[str, str] = field(default_factory=dict)  # object -> location
    closed_doors: set[str] = field(default_factory=set)

    @classmethod
    def from_dict(cls, data: dict) -> "World":
        locations = {
            name: Location(name, float(v["x"]), float(v["y"]), v.get("description", ""))
            for name, v in data["locations"].items()
        }
        connections = [Connection(*c) for c in data.get("connections", [])]
        world = cls(
            name=data["name"],
            resource=data.get("resource", f"sim://{data['name']}"),
            start=data["start"],
            locations=locations,
            connections=connections,
            objects=dict(data.get("objects") or {}),
            closed_doors=set(data.get("closed_doors") or []),
        )
        world.validate()
        return world

    @classmethod
    def from_yaml(cls, path: str | Path) -> "World":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def builtin(cls, name: str = "house") -> "World":
        return cls.from_yaml(DATA_DIR / f"{name}.yaml")

    def validate(self) -> None:
        if self.start not in self.locations:
            raise ValueError(f"start location {self.start!r} is not a known location")
        for c in self.connections:
            for end in (c.a, c.b):
                if end not in self.locations:
                    raise ValueError(f"connection references unknown location {end!r}")
        for obj, loc in self.objects.items():
            if loc not in self.locations:
                raise ValueError(f"object {obj!r} placed at unknown location {loc!r}")
        unknown = self.closed_doors - self.doors
        if unknown:
            raise ValueError(f"closed_doors references unknown doors {sorted(unknown)}")

    @property
    def doors(self) -> set[str]:
        return {c.door for c in self.connections if c.door}

    def with_overrides(
        self,
        *,
        closed_doors: list[str] | None = None,
        objects: dict[str, str | None] | None = None,
        start: str | None = None,
    ) -> "World":
        """A copy with some state changed, for evaluation scenarios. In `objects`, a
        value of None removes that object from the world."""
        world = copy.deepcopy(self)
        if closed_doors is not None:
            world.closed_doors = set(closed_doors)
        for obj, loc in (objects or {}).items():
            if loc is None:
                world.objects.pop(obj, None)
            else:
                world.objects[obj] = loc
        if start is not None:
            world.start = start
        world.validate()
        return world

    def distance(self, a: str, b: str) -> float:
        la, lb = self.locations[a], self.locations[b]
        return math.hypot(la.x - lb.x, la.y - lb.y)

    def neighbors(self, name: str) -> list[tuple[str, str | None]]:
        out = []
        for c in self.connections:
            if c.a == name:
                out.append((c.b, c.door))
            elif c.b == name:
                out.append((c.a, c.door))
        return out

    def door_between(self, a: str, b: str) -> str | None:
        for other, door in self.neighbors(a):
            if other == b:
                return door
        return None

    def shortest_path(
        self, start: str, goal: str, *, excluded_doors: set[str] | frozenset = frozenset()
    ) -> list[str] | None:
        """Dijkstra over the map by Euclidean edge length, never crossing a door in
        `excluded_doors`. Returns the node list from start to goal, or None."""
        dist = {start: 0.0}
        prev: dict[str, str] = {}
        heap = [(0.0, start)]
        while heap:
            d, node = heapq.heappop(heap)
            if node == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                return path[::-1]
            if d > dist.get(node, math.inf):
                continue
            for other, door in self.neighbors(node):
                if door and door in excluded_doors:
                    continue
                nd = d + self.distance(node, other)
                if nd < dist.get(other, math.inf):
                    dist[other] = nd
                    prev[other] = node
                    heapq.heappush(heap, (nd, other))
        return None

    def describe_map(self) -> str:
        """Static map summary for the agent's system prompt. Lists doors but
        deliberately not whether they're open: the robot can't know that from the
        static map."""
        lines = ["Locations:"]
        for loc in self.locations.values():
            lines.append(f"- {loc.name} ({loc.x:+.1f}, {loc.y:+.1f}): {loc.description}")
        lines.append("Connections (a <-> b [door]):")
        for c in self.connections:
            lines.append(f"- {c.a} <-> {c.b}" + (f" [door: {c.door}]" if c.door else " [open passage]"))
        return "\n".join(lines)
