import datetime as dt
import json
from types import SimpleNamespace

from llm_memory_core import Bundle, validate
from llm_memory_core.search.vector_backend import VectorSearchBackend

from llm_memory_agent.agent import RobotAgent
from llm_memory_agent.brain import AnthropicBrain, ScriptedBrain
from llm_memory_agent.memory import EpisodicMemory
from llm_memory_agent.robot.sim import SimRobot
from llm_memory_agent.tools import tool_definitions
from llm_memory_agent.world import World

NOW = dt.datetime(2026, 11, 10, 9, 0, tzinfo=dt.timezone.utc)


def _memory(tmp_path, world, **kw):
    return EpisodicMemory(tmp_path / "bundle", world.resource,
                          VectorSearchBackend(Bundle(tmp_path / "bundle")),
                          clock=lambda: NOW, **kw)


def test_episode_runs_tools_and_writes_memory(tmp_path):
    world = World.builtin().with_overrides(closed_doors=["lab_door_west"])
    robot, memory = SimRobot(world), _memory(tmp_path, world)
    memory.start_session("s1")
    brain = ScriptedBrain([
        ("navigate_to", {"location": "lab", "via": [], "avoid_doors": []}),
        ("navigate_to", {"location": "lab", "via": [], "avoid_doors": []}),
        ("look", {}),
        ("finish", {"success": True, "answer": "At the lab."}),
    ])
    result = RobotAgent(robot, brain, memory).run("Go to the lab.")
    memory.end_session()

    assert result.success_claimed is True and result.final_location == "lab"
    assert result.steps == 4 and len(result.trace.nav_failures) == 1
    assert result.trace.blocked_doors == [("lab_door_west", "living_room", "lab")]
    # The tool error reached the model as a tool result in the next turn.
    last_user = brain.seen_messages[1][-1]["content"][0]
    assert last_user["type"] == "tool_result" and "lab_door_west" in last_user["content"]

    types = sorted(Bundle(tmp_path / "bundle").read_concept(p).type
                   for p in Bundle(tmp_path / "bundle").walk())
    assert types == ["Environment Observation", "Failure/Recovery Note", "Mission Run", "Task Attempt"]
    # Everything written is a valid OKF bundle.
    assert [i for i in validate(Bundle(tmp_path / "bundle")) if i.severity == "error"] == []


def test_next_session_retrieves_previous_failure(tmp_path):
    world = World.builtin().with_overrides(closed_doors=["lab_door_west"])
    memory = _memory(tmp_path, world)
    memory.start_session("s1")
    RobotAgent(SimRobot(world), ScriptedBrain([
        ("navigate_to", {"location": "lab", "via": [], "avoid_doors": []}),
        ("finish", {"success": False, "answer": "Blocked."}),
    ]), memory).run("Go to the lab.")
    memory.end_session()

    memory.start_session("s2")
    brain = ScriptedBrain([("finish", {"success": True, "answer": "ok"})])
    result = RobotAgent(SimRobot(world), brain, memory).run("Drive to the lab please.")
    first_message = brain.seen_messages[0][0]["content"]
    assert "lab_door_west" in first_message and "<memory" in first_message
    assert any("failure" in p for p in result.memories_injected)


def test_memory_disabled_hides_memory_tools_and_context(tmp_path):
    world = World.builtin()
    memory = _memory(tmp_path, world, enabled=False)
    brain = ScriptedBrain([("search_memory", {"query": "x"}),
                           ("finish", {"success": True, "answer": ""})])
    agent = RobotAgent(SimRobot(world), brain, memory)
    result = agent.run("Go somewhere")
    assert "search_memory" not in [t["name"] for t in tool_definitions(False)]
    assert result.tool_calls[0]["is_error"]
    assert "no memory of earlier sessions" in agent.system_prompt()
    assert not (tmp_path / "bundle").exists()


def test_nudges_once_then_stops_and_respects_step_limit(tmp_path):
    world = World.builtin()
    brain = ScriptedBrain([])
    result = RobotAgent(SimRobot(world), brain).run("hello")
    assert result.steps == 2 and result.success_claimed is None
    assert "finish" in brain.seen_messages[1][-1]["content"]

    looping = ScriptedBrain([("look", {})] * 10)
    result = RobotAgent(SimRobot(world), looping, max_steps=3).run("look forever")
    assert result.steps == 3 and "Stopped after 3 steps" in result.answer


def test_unknown_tool_and_bad_args_are_tool_errors():
    world = World.builtin()
    brain = ScriptedBrain([("fly", {}), ("navigate_to", {"where": "x"}),
                           ("finish", {"success": False, "answer": "no"})])
    result = RobotAgent(SimRobot(world), brain).run("fly")
    assert [c["is_error"] for c in result.tool_calls] == [True, True, False]


# --------------------------------------------------------------- AnthropicBrain


class _FakeMessages:
    def __init__(self, response):
        self.response, self.calls = response, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _fake_client(response):
    msgs = _FakeMessages(response)
    return SimpleNamespace(messages=msgs, beta=SimpleNamespace(messages=msgs)), msgs


def test_anthropic_brain_request_shape_and_parsing():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""),
                 SimpleNamespace(type="text", text="Heading out."),
                 SimpleNamespace(type="tool_use", id="tu_1", name="navigate_to",
                                 input={"location": "kitchen", "via": [], "avoid_doors": []})],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                              cache_read_input_tokens=80, cache_creation_input_tokens=0),
    )
    client, msgs = _fake_client(response)
    brain = AnthropicBrain(client=client)
    tools = tool_definitions(True)
    turn = brain.step("sys", tools, [{"role": "user", "content": "go"}])

    req = msgs.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": "medium"}
    assert req["fallbacks"] == "default" and req["betas"] == [AnthropicBrain.FALLBACK_BETA]
    assert req["cache_control"] == {"type": "ephemeral"}
    assert "tool_choice" not in req  # forced tool choice is rejected on this model
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in tools)

    assert [c.name for c in turn.tool_calls] == ["navigate_to"]
    assert turn.text == "Heading out." and turn.assistant_content is response.content
    assert turn.usage["cache_read_input_tokens"] == 80


def test_anthropic_brain_without_fallbacks_uses_plain_endpoint():
    response = SimpleNamespace(content=[], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    client, msgs = _fake_client(response)
    AnthropicBrain(client=client, use_fallbacks=False, effort="low").step("s", [], [])
    assert "fallbacks" not in msgs.calls[0] and msgs.calls[0]["output_config"] == {"effort": "low"}


def test_tool_results_are_json():
    from llm_memory_agent.tools import ToolExecutor

    out = ToolExecutor(SimRobot(World.builtin()), None).execute("get_robot_state", {})
    assert json.loads(out.content)["location"] == "dock"
