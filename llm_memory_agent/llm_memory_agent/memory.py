"""EpisodicMemory: the agent's long-term memory. It writes what happened as OKF
concepts and retrieves them semantically (RAG).

Writes go to the OKF bundle (source of truth, plain markdown):
- one Mission Run per agent session,
- one Task Attempt per operator command (command, outcome, answer, route),
- one Failure/Recovery Note per failed navigation (e.g. a closed door),
- one Environment Observation per `look` (what objects were where, and when),
- Operator Notes for facts the operator states ("my desk" means the office).

Reads go through llm_memory_core's VectorSearchBackend (Chroma + MiniLM by
default), re-indexed incrementally after every write. Each memory text carries its
own timestamp, so the agent can tell old observations from new ones.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from llm_memory_core import (
    Bundle,
    Concept,
    EnvironmentObservation,
    FailureNote,
    MissionRun,
    TaskAttempt,
)
from llm_memory_core.search.vector_backend import VectorSearchBackend

GENERATED_BY = "llm_memory_agent/0.1.0"


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _stamp(at: dt.datetime) -> str:
    return at.strftime("%Y-%m-%d %H:%M UTC")


@dataclass
class MemoryHit:
    path: str
    text: str
    score: float | None


@dataclass
class EpisodeRecord:
    """What the agent did for one operator command; written as a Task Attempt."""

    command: str
    success_claimed: bool | None
    answer: str
    route: list[str] = field(default_factory=list)
    nav_failures: list[str] = field(default_factory=list)  # human-readable messages
    blocked_doors: list[tuple[str, str, str]] = field(default_factory=list)  # (door, a, b)
    observations: list[tuple[str, list[str]]] = field(default_factory=list)  # (loc, objects)


class EpisodicMemory:
    def __init__(
        self,
        bundle_root: str | Path,
        environment_resource: str,
        search: VectorSearchBackend | None = None,
        *,
        enabled: bool = True,
        clock: Callable[[], dt.datetime] = _utcnow,
    ):
        self.bundle = Bundle(bundle_root)
        self.environment_resource = environment_resource
        self.search_backend = search or VectorSearchBackend(self.bundle)
        if self.search_backend.bundle.root != self.bundle.root:
            raise ValueError("search backend must index the same bundle as the memory")
        self.enabled = enabled
        self.clock = clock
        self._mission_dir: Path | None = None
        self._mission: MissionRun | None = None
        self._counters: dict[str, int] = {}

    # ------------------------------------------------------------------ sessions

    def start_session(self, name: str = "agent") -> Path:
        at = self.clock()
        self.bundle.ensure_root_index()
        self._mission_dir = Path("missions") / f"{at.strftime('%Y-%m-%dT%H-%M-%SZ')}-{name}"
        self._counters = {}
        self._mission = MissionRun(
            title=f"Agent session {name} ({_stamp(at)})",
            resource=self.environment_resource,
            status="draft",
            generated={"by": GENERATED_BY, "at": at.isoformat()},
            body=f"Agent session started {_stamp(at)} in {self.environment_resource}.\n",
        )
        self.bundle.write_concept(self._mission_dir / "mission.md", self._mission)
        self.bundle.append_log(self._mission_dir, "Session started.", at=at)
        return self._mission_dir

    def end_session(self) -> None:
        if self._mission is None:
            return
        self._mission.status = "stable"
        self._mission.body += f"Session ended {_stamp(self.clock())}.\n"
        self.bundle.write_concept(self._mission_dir / "mission.md", self._mission)
        self._mission = None
        self.reindex()

    def _next_path(self, kind: str, subdir: str) -> Path:
        if self._mission_dir is None:
            self.start_session()
        self._counters[kind] = self._counters.get(kind, 0) + 1
        return self._mission_dir / subdir / f"{kind}-{self._counters[kind]:03d}.md"

    # -------------------------------------------------------------------- writes

    def _write(self, kind: str, subdir: str, concept: Concept) -> Path:
        path = self._next_path(kind, subdir)
        self.bundle.write_concept(path, concept)
        return path

    def record_observation(self, location: str, objects: list[str]) -> Path:
        at = self.clock()
        seen = ", ".join(objects) if objects else "no notable objects"
        return self._write("observation", "events", EnvironmentObservation(
            title=f"Seen at {location}",
            resource=self.environment_resource,
            tags=["observation", location, *objects],
            generated={"by": GENERATED_BY, "at": at.isoformat()},
            body=f"Observed {_stamp(at)} at {location}: {seen}.\n",
        ))

    def record_failure(self, message: str, *, door: str | None = None,
                       between: tuple[str, str] | None = None) -> Path:
        at = self.clock()
        body = f"{_stamp(at)}: {message}\n"
        if door and between:
            body += (f"\nDoor {door} (between {between[0]} and {between[1]}) was closed. "
                     f"Next time, consider routing around it with avoid_doors=[\"{door}\"].\n")
        return self._write("failure", "events", FailureNote(
            title=f"Navigation blocked at {door}" if door else "Navigation failure",
            resource=self.environment_resource,
            tags=["navigation", "failure"] + ([door] if door else []),
            generated={"by": GENERATED_BY, "at": at.isoformat()},
            body=body,
        ))

    def record_note(self, text: str) -> Path:
        at = self.clock()
        return self._write("note", "notes", Concept(
            type="Operator Note",
            title="Operator note",
            resource=self.environment_resource,
            tags=["operator"],
            generated={"by": GENERATED_BY, "at": at.isoformat()},
            body=f"Noted {_stamp(at)}: {text}\n",
        ))

    def record_episode(self, ep: EpisodeRecord) -> Path:
        """Write a Task Attempt plus its failure/observation events, then re-index."""
        at = self.clock()
        for door, a, b in ep.blocked_doors:
            self.record_failure(
                f"Navigation from {a} towards {b} aborted: door {door} closed.",
                door=door, between=(a, b))
        for msg in ep.nav_failures:
            if not any(door in msg for door, _, _ in ep.blocked_doors):
                self.record_failure(msg)
        for loc, objects in ep.observations:
            self.record_observation(loc, objects)

        outcome = {True: "reported success", False: "reported failure", None: "no report"}[
            ep.success_claimed]
        body = (f"Command at {_stamp(at)}: \"{ep.command}\"\n\n"
                f"Outcome: {outcome}. {ep.answer}\n\n"
                f"Route: {' -> '.join(ep.route) if ep.route else '(did not move)'}\n")
        path = self._write("task", "tasks", TaskAttempt(
            title=ep.command[:80],
            resource=self.environment_resource,
            status="stable",
            tags=["task"],
            generated={"by": GENERATED_BY, "at": at.isoformat()},
            body=body,
        ))
        if self._mission_dir is not None:
            self.bundle.append_log(self._mission_dir, f"Task: {ep.command} -> {outcome}", at=at)
        self.reindex()
        return path

    def reindex(self) -> None:
        self.search_backend.index()

    # --------------------------------------------------------------------- reads

    def retrieve(self, query: str, k: int = 5) -> list[MemoryHit]:
        """Top-k memories for `query` in this environment, one per document."""
        if not self.enabled:
            return []
        results = self.search_backend.search(query, k=k * 2, resource=self.environment_resource)
        hits, seen = [], set()
        for r in results:
            key = r.path.as_posix()
            if key in seen or key.endswith("mission.md"):
                continue
            seen.add(key)
            hits.append(MemoryHit(key, r.snippet.strip(), r.score))
            if len(hits) == k:
                break
        return hits


def open_memory(
    bundle_root: str | Path,
    environment_resource: str,
    *,
    index_dir: str | Path | None = None,
    embedder: str = "minilm",
    enabled: bool = True,
) -> EpisodicMemory:
    """Memory with a persistent Chroma index next to (not inside) the bundle, so the
    bundle stays clean, portable markdown. Default index dir: `<bundle_root>.index`."""
    from llm_memory_core.search.vector_backend import (
        ChromaVectorStore,
        HashingEmbedder,
        MiniLMEmbedder,
    )

    bundle_root = Path(bundle_root).expanduser()
    index_dir = Path(index_dir).expanduser() if index_dir else bundle_root.with_name(
        bundle_root.name + ".index")
    emb = MiniLMEmbedder() if embedder == "minilm" else HashingEmbedder()
    # Collection name includes the embedder, so switching embedders can never mix
    # vectors from two models in one collection.
    store = ChromaVectorStore(index_dir, collection=f"okf_{emb.name}".replace(".", "_")[:60])
    search = VectorSearchBackend(Bundle(bundle_root), store, emb)
    memory = EpisodicMemory(bundle_root, environment_resource, search, enabled=enabled)
    memory.reindex()  # pick up anything written while we weren't running
    return memory
