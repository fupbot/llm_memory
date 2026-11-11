"""Evaluation tasks: loading, memory seeding, and grading against simulator ground
truth. Grading never trusts the agent's own report: "final location" is the
simulator's state and "visited" is the simulator's trajectory. `success_claimed` is
checked separately, so that claiming success on a failed task (a false success claim)
is caught as an error."""
from __future__ import annotations

import datetime as dt
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..agent import EpisodeResult
from ..memory import EpisodicMemory
from ..world import DATA_DIR, World

# Reference "now" for seeded memories, so memory text (and thus retrieval and the
# prompt) is identical across runs.
EVAL_NOW = dt.datetime(2026, 11, 10, 9, 0, tzinfo=dt.timezone.utc)

_FILLER_OBJECTS = ["umbrella", "backpack", "shoe rack", "cardboard box", "water bottle",
                   "vacuum cleaner", "blanket", "board game", "trash bin", "lamp"]


@dataclass
class Task:
    id: str
    category: str
    command: str
    expect: dict
    world: dict = field(default_factory=dict)
    memory_seed: list[dict] = field(default_factory=list)
    needs_memory: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            id=d["id"], category=d["category"], command=d["command"],
            expect=d.get("expect") or {}, world=d.get("world") or {},
            memory_seed=d.get("memory_seed") or [], needs_memory=bool(d.get("needs_memory")),
        )

    def build_world(self, base: World) -> World:
        return base.with_overrides(**self.world)


def load_tasks(path: str | Path | None = None) -> list[Task]:
    path = Path(path) if path else DATA_DIR / "tasks.yaml"
    tasks = [Task.from_dict(d) for d in yaml.safe_load(path.read_text(encoding="utf-8"))]
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids in task file")
    return tasks


class _FixedClock:
    def __init__(self, at: dt.datetime):
        self.at = at

    def __call__(self) -> dt.datetime:
        return self.at


def seed_memory(memory: EpisodicMemory, task: Task, world: World) -> None:
    """Write the task's prior experience into memory as past sessions, oldest first.
    Restores the memory's clock afterwards."""
    original_clock = memory.clock
    rng = random.Random(task.id)  # deterministic distractors per task
    entries: list[tuple[float, dict]] = []
    for item in task.memory_seed:
        if "distractors" in item:
            locations = list(world.locations)
            for _ in range(int(item["distractors"])):
                days = rng.uniform(5, 60)
                if rng.random() < 0.5:
                    entries.append((days, {"observation": {
                        "location": rng.choice(locations),
                        "objects": rng.sample(_FILLER_OBJECTS, 2)}}))
                else:
                    a, b = rng.sample(locations, 2)
                    entries.append((days, {"trip": (a, b)}))
        else:
            entries.append((float(item.get("days_ago", 1)), item))

    try:
        for n, (days_ago, item) in enumerate(sorted(entries, key=lambda e: -e[0])):
            memory.clock = _FixedClock(EVAL_NOW - dt.timedelta(days=days_ago))
            memory.start_session(f"seed{n:03d}")  # unique dir even for equal timestamps
            if "observation" in item:
                obs = item["observation"]
                memory.record_observation(obs["location"], list(obs["objects"]))
            elif "failure" in item:
                f = item["failure"]
                a, b = f["between"]
                memory.record_failure(
                    f"Navigation from {a} towards {b} aborted: door {f['door']} closed.",
                    door=f["door"], between=(a, b))
            elif "note" in item:
                memory.record_note(item["note"])
            elif "trip" in item:
                a, b = item["trip"]
                memory.record_note(f"Routine patrol from {a} to {b} completed without issues.")
            memory.end_session()
    finally:
        memory.clock = original_clock
    memory.reindex()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("_", " ")).strip()


def _contains(haystack: str, needle: str) -> bool:
    return re.search(r"\b" + re.escape(_norm(needle)) + r"\b", _norm(haystack)) is not None


def _is_subsequence(needed: list[str], seq: list[str]) -> bool:
    it = iter(seq)
    return all(any(x == n for x in it) for n in needed)


@dataclass
class Grade:
    passed: bool
    checks: dict[str, bool]
    false_success_claim: bool


def grade(task: Task, result: EpisodeResult, trajectory: list[str]) -> Grade:
    """Grade an episode. `trajectory` is the simulator's full list of locations the
    robot was in, starting with the start location."""
    exp = task.expect
    checks: dict[str, bool] = {}
    if "final_location" in exp:
        allowed = exp["final_location"]
        allowed = [allowed] if isinstance(allowed, str) else list(allowed)
        checks["final_location"] = result.final_location in allowed
    if "visited_in_order" in exp:
        checks["visited_in_order"] = _is_subsequence(list(exp["visited_in_order"]), trajectory[1:])
    if "not_visited" in exp:
        checks["not_visited"] = not set(exp["not_visited"]) & set(trajectory)
    if "answer_contains_any" in exp:
        checks["answer_contains_any"] = any(_contains(result.answer, s)
                                            for s in exp["answer_contains_any"])
    if "answer_contains_all" in exp:
        checks["answer_contains_all"] = all(_contains(result.answer, s)
                                            for s in exp["answer_contains_all"])
    if "success_claimed" in exp:
        checks["success_claimed"] = result.success_claimed is exp["success_claimed"]
    checks["no_error"] = result.error is None

    passed = all(checks.values())
    world_checks = [v for k, v in checks.items() if k not in ("success_claimed", "no_error")]
    # A false success claim: the agent said it succeeded, but either the task expected
    # it to report failure, or the world state shows it didn't do what was asked.
    false_claim = result.success_claimed is True and (
        exp.get("success_claimed") is False or not all(world_checks))
    return Grade(passed, checks, false_claim)
