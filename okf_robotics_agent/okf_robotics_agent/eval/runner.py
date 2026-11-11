"""okf_robotics_eval: run the task set against an agent brain, with long-term memory
ON and/or OFF, and report task success and efficiency.

    okf_robotics_eval --brain anthropic --memory both --trials 3
    okf_robotics_eval --brain baseline            # no API key needed

Every (task, memory mode, trial) runs in a fresh temporary bundle and vector index,
so nothing leaks between episodes. Writes results.jsonl (one line per episode,
including the full tool trace) and summary.md to --out.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from okf_robotics_core.search.vector_backend import (
    ChromaVectorStore,
    HashingEmbedder,
    InMemoryVectorStore,
    MiniLMEmbedder,
    VectorSearchBackend,
)

from ..agent import RobotAgent
from ..brain import AnthropicBrain, Brain, KeywordBaselineBrain
from ..memory import EpisodicMemory
from ..robot.sim import SimRobot
from ..world import World
from .tasks import EVAL_NOW, Task, grade, load_tasks, seed_memory


@dataclass
class RunConfig:
    brain: str = "anthropic"
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    store: str = "chroma"
    embedder: str = "minilm"
    max_steps: int = 20


def make_brain(cfg: RunConfig, world: World, client=None) -> Brain:
    if cfg.brain == "baseline":
        return KeywordBaselineBrain(list(world.locations))
    if cfg.brain == "anthropic":
        return AnthropicBrain(cfg.model, effort=cfg.effort, client=client)
    raise ValueError(f"unknown brain {cfg.brain!r}")


def run_episode(task: Task, memory_on: bool, trial: int, cfg: RunConfig,
                base_world: World, embedder, client=None) -> dict:
    world = task.build_world(base_world)
    robot = SimRobot(world)
    with tempfile.TemporaryDirectory(prefix="okf_eval_") as tmp:
        memory = None
        if memory_on:
            bundle_root = Path(tmp) / "bundle"
            store = (ChromaVectorStore(Path(tmp) / "index") if cfg.store == "chroma"
                     else InMemoryVectorStore())
            from okf_robotics_core import Bundle

            search = VectorSearchBackend(Bundle(bundle_root), store, embedder)
            memory = EpisodicMemory(bundle_root, world.resource, search,
                                    clock=lambda: EVAL_NOW)
            seed_memory(memory, task, world)
            memory.start_session("eval")
        agent = RobotAgent(robot, make_brain(cfg, world, client), memory, max_steps=cfg.max_steps)
        result = agent.run(task.command)
        if memory is not None:
            memory.end_session()

    g = grade(task, result, robot.trajectory)
    return {
        "task": task.id, "category": task.category, "needs_memory": task.needs_memory,
        "memory": memory_on, "trial": trial, "passed": g.passed, "checks": g.checks,
        "false_success_claim": g.false_success_claim,
        "success_claimed": result.success_claimed, "answer": result.answer,
        "final_location": result.final_location, "trajectory": robot.trajectory,
        "steps": result.steps, "nav_failures": len(result.trace.nav_failures),
        "distance_m": round(result.trace.distance, 2), "usage": result.usage,
        "memories_injected": result.memories_injected, "tool_calls": result.tool_calls,
        "error": result.error,
    }


def _rate(rows: list[dict]) -> str:
    if not rows:
        return "-"
    passed = sum(r["passed"] for r in rows)
    return f"{100 * passed / len(rows):.0f}% ({passed}/{len(rows)})"


def _mean(rows: list[dict], key: str) -> str:
    return f"{statistics.mean(r[key] for r in rows):.2f}" if rows else "-"


def summarize(rows: list[dict], header: str) -> str:
    modes = sorted({r["memory"] for r in rows}, reverse=True)
    label = {True: "memory ON", False: "memory OFF"}
    cols = [label[m] for m in modes]
    out = [header, "", "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]

    def row(name, fn):
        out.append(f"| {name} | " + " | ".join(fn([r for r in rows if r["memory"] == m])
                                               for m in modes) + " |")

    row("**Task success (all)**", _rate)
    for cat in sorted({r["category"] for r in rows}):
        row(f"  {cat}", lambda rs, c=cat: _rate([r for r in rs if r["category"] == c]))
    row("Tasks that need memory", lambda rs: _rate([r for r in rs if r["needs_memory"]]))
    row("Tasks that don't need memory", lambda rs: _rate([r for r in rs if not r["needs_memory"]]))
    row("False success claims", lambda rs: str(sum(r["false_success_claim"] for r in rs)))
    row("Mean nav failures / episode", lambda rs: _mean(rs, "nav_failures"))
    row("Mean distance (m) / episode", lambda rs: _mean(rs, "distance_m"))
    row("Mean agent steps / episode", lambda rs: _mean(rs, "steps"))
    row("Errors", lambda rs: str(sum(r["error"] is not None for r in rs)))
    row("Input tokens (total)", lambda rs: str(sum(r["usage"].get("input_tokens", 0) for r in rs)))
    row("Cache-read tokens (total)",
        lambda rs: str(sum(r["usage"].get("cache_read_input_tokens", 0) for r in rs)))
    row("Output tokens (total)", lambda rs: str(sum(r["usage"].get("output_tokens", 0) for r in rs)))

    mem_tasks = sorted({r["task"] for r in rows if r["category"] == "memory"})
    if mem_tasks and len(modes) == 2:
        out += ["", "Memory tasks, per task (pass rate / mean nav failures / mean distance m):", "",
                "| task | memory ON | memory OFF |", "|---|---|---|"]
        for t in mem_tasks:
            cells = []
            for m in modes:
                rs = [r for r in rows if r["task"] == t and r["memory"] == m]
                cells.append(f"{_rate(rs)} / {_mean(rs, 'nav_failures')} / {_mean(rs, 'distance_m')}")
            out.append(f"| {t} | " + " | ".join(cells) + " |")

    failed = [r for r in rows if not r["passed"]]
    if failed:
        out += ["", "Failed episodes:", ""]
        for r in failed:
            bad = [k for k, v in r["checks"].items() if not v]
            out.append(f"- `{r['task']}` ({label[r['memory']]}, trial {r['trial']}): failed "
                       f"{', '.join(bad)}; answer: {r['answer'][:120]!r}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--brain", choices=["anthropic", "baseline"], default="anthropic")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--memory", choices=["on", "off", "both"], default="both")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--tasks", nargs="*", help="only these task ids")
    p.add_argument("--task-file", help="alternative tasks YAML")
    p.add_argument("--store", choices=["chroma", "memory"], default="chroma")
    p.add_argument("--embedder", choices=["minilm", "hashing"], default="minilm")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="eval_results")
    args = p.parse_args(argv)

    cfg = RunConfig(args.brain, args.model, args.effort, args.store, args.embedder, args.max_steps)
    tasks = load_tasks(args.task_file)
    if args.tasks:
        tasks = [t for t in tasks if t.id in set(args.tasks)]
    modes = {"on": [True], "off": [False], "both": [True, False]}[args.memory]
    base_world = World.builtin("house")
    embedder = MiniLMEmbedder() if args.embedder == "minilm" else HashingEmbedder()
    client = None
    if cfg.brain == "anthropic":
        import anthropic

        client = anthropic.Anthropic()

    jobs = [(t, m, n) for t in tasks for m in modes for n in range(args.trials)]
    print(f"Running {len(jobs)} episodes ({len(tasks)} tasks x {len(modes)} memory modes x "
          f"{args.trials} trials) with {cfg.brain}"
          + (f" {cfg.model} effort={cfg.effort}" if cfg.brain == "anthropic" else ""), file=sys.stderr)

    def work(job):
        task, mem, trial = job
        row = run_episode(task, mem, trial, cfg, base_world, embedder, client)
        print(f"  {'PASS' if row['passed'] else 'FAIL'} {task.id} memory={'on' if mem else 'off'} "
              f"trial={trial}" + (f" error={row['error']}" if row["error"] else ""), file=sys.stderr)
        return row

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(work, jobs))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    who = cfg.brain if cfg.brain != "anthropic" else f"{cfg.model} (effort {cfg.effort})"
    header = (f"## okf_robotics_agent eval: {who}\n\n{len(tasks)} tasks x {args.trials} trial(s), "
              f"store={cfg.store}, embedder={cfg.embedder}, run {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC")
    summary = summarize(rows, header)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
