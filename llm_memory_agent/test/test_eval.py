import tempfile
from pathlib import Path

from llm_memory_core import Bundle
from llm_memory_core.search.vector_backend import HashingEmbedder, VectorSearchBackend

from llm_memory_agent.agent import EpisodeResult
from llm_memory_agent.eval.runner import RunConfig, run_episode, summarize
from llm_memory_agent.eval.tasks import EVAL_NOW, Task, grade, load_tasks, seed_memory
from llm_memory_agent.memory import EpisodicMemory
from llm_memory_agent.world import World


def _result(**kw):
    base = dict(command="c", success_claimed=True, answer="", steps=1, final_location="dock")
    base.update(kw)
    return EpisodeResult(**base)


def test_task_file_is_valid_against_world():
    world = World.builtin()
    tasks = load_tasks()
    assert len(tasks) >= 20
    for t in tasks:
        t.build_world(world)  # overrides reference real doors/locations
        for key in ("final_location", "visited_in_order", "not_visited"):
            vals = t.expect.get(key, [])
            for v in ([vals] if isinstance(vals, str) else vals):
                assert v in world.locations, (t.id, v)


def test_grade_checks_world_state_not_agent_claims():
    task = Task("t", "navigation", "go", {"final_location": "kitchen", "success_claimed": True})
    ok = grade(task, _result(final_location="kitchen"), ["dock", "hallway", "kitchen"])
    assert ok.passed and not ok.false_success_claim
    lie = grade(task, _result(final_location="hallway"), ["dock", "hallway"])
    assert not lie.passed and lie.false_success_claim


def test_grade_honesty_and_answer_matching():
    task = Task("t", "failure_handling", "go to roof",
                {"success_claimed": False, "answer_contains_any": ["lab_door_north"]})
    g = grade(task, _result(success_claimed=False, answer="The lab door north was closed."), ["dock"])
    assert g.passed
    g = grade(task, _result(success_claimed=True, answer="lab_door_north"), ["dock"])
    assert not g.passed and g.false_success_claim
    # Word-boundary matching: "no" must not match inside "know".
    t2 = Task("t2", "x", "c", {"answer_contains_any": ["no"]})
    assert not grade(t2, _result(answer="I know"), ["dock"]).passed


def test_grade_visit_order_and_not_visited():
    task = Task("t", "x", "c", {"visited_in_order": ["garage", "lab"], "not_visited": ["office"]})
    assert grade(task, _result(), ["dock", "garage", "dock", "hallway", "living_room", "lab"]).passed
    assert not grade(task, _result(), ["dock", "hallway", "living_room", "lab", "dock", "garage"]).passed
    assert not grade(task, _result(), ["dock", "garage", "office", "lab"]).passed


def test_seed_memory_is_deterministic_and_retrievable():
    task = next(t for t in load_tasks() if t.id == "mem_needle_in_haystack")
    world = task.build_world(World.builtin())
    listings = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as d:
            mem = EpisodicMemory(Path(d), world.resource,
                                 VectorSearchBackend(Bundle(Path(d)), embedder=HashingEmbedder()),
                                 clock=lambda: EVAL_NOW)
            seed_memory(mem, task, world)
            listings.append(sorted(p.as_posix() for p in mem.bundle.walk()))
            top = mem.retrieve(task.command, 3)
            assert "charger_cable" in top[0].text
    assert listings[0] == listings[1]
    assert len(listings[0]) == 41 * 2  # 41 seed sessions x (mission + one event)


def test_run_episode_baseline_end_to_end():
    tasks = {t.id: t for t in load_tasks()}
    cfg = RunConfig(brain="baseline", store="memory", embedder="hashing")
    world, emb = World.builtin(), HashingEmbedder()
    row = run_episode(tasks["nav_round_trip"], True, 0, cfg, world, emb)
    assert row["passed"] and row["trajectory"][-1] == "dock"
    row = run_episode(tasks["reroute_closed_door"], False, 0, cfg, world, emb)
    assert not row["passed"] and row["false_success_claim"] and row["nav_failures"] == 1

    md = summarize([row], "## test")
    assert "False success claims | 1" in md and "reroute_closed_door" in md
