import json

import pytest

from llm_memory_agent.agent import RobotAgent
from llm_memory_agent.brain import (
    AnthropicBrain,
    LLMRequestError,
    OpenAICompatBrain,
    make_brain,
    to_openai_messages,
)
from llm_memory_agent.robot.sim import SimRobot
from llm_memory_agent.tools import tool_definitions
from llm_memory_agent.world import World


class FakeServer:
    """Stands in for an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, url, payload, headers, timeout):
        self.requests.append({"url": url, "payload": json.loads(json.dumps(payload)),
                              "headers": headers})
        return self.replies.pop(0)


def _reply(content=None, tool_calls=None, finish="stop", usage=None):
    return {"choices": [{"message": {"role": "assistant", "content": content,
                                     "tool_calls": tool_calls},
                         "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5}}


def _call(name, args, id="c1", as_string=True):
    return {"id": id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args) if as_string else args}}


def _brain(server, **kw):
    return OpenAICompatBrain("qwen3:8b", base_url="http://localhost:11434/v1/", post=server, **kw)


def test_request_shape_and_tool_call_parsing():
    server = FakeServer([_reply("<think>hmm</think>Going.",
                                [_call("navigate_to", {"location": "kitchen", "via": [],
                                                       "avoid_doors": []})], "tool_calls")])
    turn = _brain(server).step("sys", tool_definitions(True), [{"role": "user", "content": "go"}])

    req = server.requests[0]
    assert req["url"] == "http://localhost:11434/v1/chat/completions"
    assert req["headers"] == {}  # Ollama: no key
    p = req["payload"]
    assert p["model"] == "qwen3:8b"
    assert p["messages"][:2] == [{"role": "system", "content": "sys"},
                                 {"role": "user", "content": "go"}]
    fn = p["tools"][0]["function"]
    assert p["tools"][0]["type"] == "function" and fn["name"] == "navigate_to"
    assert fn["parameters"]["required"] == ["location", "via", "avoid_doors"]

    assert turn.text == "Going."  # <think> stripped
    assert turn.stop_reason == "tool_use"
    assert [(c.id, c.name, c.input["location"]) for c in turn.tool_calls] == [
        ("c1", "navigate_to", "kitchen")]
    assert turn.usage == {"input_tokens": 10, "output_tokens": 5}


def test_lenient_parsing_of_local_model_output():
    server = FakeServer([_reply(None, [
        {"function": {"name": "look", "arguments": ""}},                     # no id, empty args
        _call("get_robot_state", {}, id="", as_string=False),               # dict args
        {"id": "x", "function": {"name": "finish", "arguments": "{oops"}},  # bad JSON
    ])])
    turn = _brain(server).step("s", [], [{"role": "user", "content": "go"}])
    assert "tools" not in server.requests[0]["payload"]
    ids = [c.id for c in turn.tool_calls]
    assert len(set(ids)) == 3 and all(ids)
    assert turn.tool_calls[0].input == {} and turn.tool_calls[1].input == {}
    assert turn.tool_calls[2].input == {"arguments_were_not_valid_json": "{oops"}


@pytest.mark.parametrize("finish,expected", [("stop", "end_turn"), ("length", "max_tokens"),
                                             ("content_filter", "refusal")])
def test_finish_reason_mapping(finish, expected):
    turn = _brain(FakeServer([_reply("hi", None, finish)])).step("s", [], [])
    assert turn.stop_reason == expected and not turn.tool_calls


def test_history_conversion():
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "a", "name": "look", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "{}", "is_error": False},
            {"type": "tool_result", "tool_use_id": "b", "content": "bad", "is_error": True}]},
        {"role": "user", "content": "Call `finish`."},
    ]
    out = to_openai_messages("sys", msgs)
    assert out[2] == {"role": "assistant", "content": "ok", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "look", "arguments": "{}"}}]}
    assert out[3] == {"role": "tool", "tool_call_id": "a", "content": "{}"}
    assert out[4] == {"role": "tool", "tool_call_id": "b", "content": "ERROR: bad"}
    assert out[5] == {"role": "user", "content": "Call `finish`."}


def test_server_error_is_raised_and_recorded_by_agent():
    server = FakeServer([{"error": {"message": "model 'nope' not found"}}])
    result = RobotAgent(SimRobot(World.builtin()), _brain(server)).run("go to the kitchen")
    assert result.error and "not found" in result.error


def test_full_episode_through_agent_loop():
    server = FakeServer([
        _reply(None, [_call("navigate_to", {"location": "kitchen", "via": [],
                                            "avoid_doors": []}, id="n1")], "tool_calls"),
        _reply(None, [_call("finish", {"success": True, "answer": "In the kitchen."},
                            id="f1")], "tool_calls"),
    ])
    robot = SimRobot(World.builtin())
    result = RobotAgent(robot, _brain(server, api_key="k")).run("Go to the kitchen.")
    assert result.success_claimed is True and result.final_location == "kitchen"
    assert result.usage == {"input_tokens": 20, "output_tokens": 10}
    second = server.requests[1]
    assert second["headers"] == {"Authorization": "Bearer k"}
    roles = [m["role"] for m in second["payload"]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert json.loads(second["payload"]["messages"][3]["content"])["location"] == "kitchen"


# ------------------------------------------------------------------ make_brain


def test_make_brain_providers(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    b = make_brain("ollama")
    assert isinstance(b, OpenAICompatBrain) and b.name == "ollama:qwen3:8b"
    assert b.base_url == "http://localhost:11434/v1" and b.api_key is None

    monkeypatch.setenv("OLLAMA_HOST", "gpu-box")
    assert make_brain("ollama", "llama3.1:8b").base_url == "http://gpu-box:11434/v1"
    monkeypatch.setenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    assert make_brain("ollama").base_url == "http://host.docker.internal:11434/v1"
    assert make_brain("ollama", base_url="http://x:1/v1").base_url == "http://x:1/v1"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    o = make_brain("openai", "gpt-5-mini")
    assert o.base_url == "https://api.openai.com/v1" and o.api_key == "sk-test"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    assert make_brain("openai", "m").base_url == "https://openrouter.ai/api/v1"

    a = make_brain("anthropic", client=object())
    assert isinstance(a, AnthropicBrain) and a.model == "claude-opus-5-5"


def test_make_brain_errors(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="model"):
        make_brain("openai")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_brain("openai", "gpt-5-mini")
    with pytest.raises(ValueError, match="unknown provider"):
        make_brain("gemini")


def test_unreachable_server_message():
    brain = OpenAICompatBrain("m", base_url="http://127.0.0.1:9/v1", timeout=2)
    with pytest.raises(LLMRequestError, match="is the server running"):
        brain.step("s", [], [{"role": "user", "content": "hi"}])
