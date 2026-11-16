# robot_ai_memory

[![CI](https://github.com/fupbot/robot_ai_memory/actions/workflows/ci.yml/badge.svg)](https://github.com/fupbot/robot_ai_memory/actions/workflows/ci.yml)

Persistent, portable memory for ROS 2 robots and the LLMs that drive them.

Robots controlled by LLM planners forget everything between sessions: the door that
was closed yesterday, where the toolbox was last seen, what the operator means by "my
desk". This repo stores that memory as an [Open Knowledge Format][okf] (OKF) bundle:
plain markdown with YAML frontmatter, readable with `cat`, diffable in git, and usable
by any OKF-compatible tool. It isn't locked to one agent framework.

It also includes an **LLM agent that controls a simulated robot from natural-language
commands**, uses that memory through retrieval-augmented generation (vector search
over the bundle), and comes with an evaluation set that measures how often it
completes tasks correctly, with and without memory.

## Packages

| Package | What it does | Needs ROS? | Needs an API key? |
|---|---|---|---|
| `llm_memory_core` | OKF bundle read/write, validation, grep + vector search | no | no |
| `llm_memory_capture` | ROS 2 node that writes task/replan/failure events to the bundle, non-blocking | yes | no |
| `llm_memory_mcp` | MCP server: query the bundle from any LLM client | no | no |
| `llm_memory_agent` | LLM agent (Claude, a local Ollama model, or any OpenAI-compatible API): natural language → Nav2, RAG memory, simulator, eval | optional | yes |

## The agent

```
operator: "Go to the lab."
        │
        ▼
  retrieve top-k memories (Chroma, MiniLM embeddings)  ◄── OKF bundle (markdown)
        │   "2026-11-09: door lab_door_west closed"            ▲
        ▼                                                      │
  Claude tool-use loop                                         │ write episode back
    navigate_to(lab, avoid_doors=[lab_door_west])              │ (task, failures,
    look()                                                     │  observations)
    finish(success=true, "At the lab.")  ──────────────────────┘
        │
        ▼
  RobotInterface ── SimRobot (built-in, for eval)
                 └─ Nav2Robot (NavigateToPose → Gazebo or a real robot)
```

- **Coarse actions, not velocities.** The LLM picks named destinations; Nav2 (or the
  simulator) handles motion control, so model latency never sits inside a control
  loop.
- **The OKF bundle is the source of truth.** The vector index is a disposable cache
  next to it, rebuilt incrementally from the markdown.
- **Honesty is graded.** The agent must report success or failure; the eval checks
  that report against the simulator's ground truth and counts *false success claims*
  separately.

### Try it (no ROS needed)

```bash
pip install -e llm_memory_core -e llm_memory_agent -e llm_memory_mcp mcp
export ANTHROPIC_API_KEY=...   # or use a local model: see "Choosing the LLM" below

# Session 1: the west lab door is closed. The agent should hit it, reroute, and the
# failure is written to memory.
llm_memory_agent_sim -c "Go to the lab." --closed-door lab_door_west
# Session 2: the failure is retrieved, so it can plan around the door from the start.
llm_memory_agent_sim -c "Go to the lab." --closed-door lab_door_west
# Memory lives here, as markdown:
find ~/llm_memory_bundle -name '*.md' | head
```

### Choosing the LLM

`--provider` (`--brain` for `llm_memory_eval`, `provider:=` for the ROS launch file)
picks the model backend. API keys are read from the environment, never from flags.

| provider | what | needs | default model |
|---|---|---|---|
| `anthropic` (default) | Claude via the Anthropic API | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |
| `ollama` | a local model served by Ollama | Ollama running; `OLLAMA_HOST` or `--base-url` if it's not on `localhost:11434` | `qwen3:8b` |
| `openai` | any OpenAI-compatible cloud API (OpenAI, OpenRouter, Groq, ...) | `OPENAI_API_KEY`; `OPENAI_BASE_URL` or `--base-url` for non-OpenAI hosts; `--model` | none |

```bash
llm_memory_agent_sim --provider ollama --model llama3.1:8b -c "Go to the kitchen."
OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY=... \
  llm_memory_eval --brain openai --model qwen/qwen3-32b --memory both
```

Local models need tool calling support (Qwen3, Llama 3.1+, Mistral Small, ...);
larger ones follow the tool protocol more reliably. Ollama's default context window
can be too small for the system prompt, memories and a multi-step tool loop: start
the server with e.g. `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`. In Docker
(`jazzy-sim`), `OLLAMA_HOST` defaults to the host machine's Ollama.

### Evaluation

```bash
llm_memory_eval --memory both --trials 3        # Claude, memory ON vs OFF
llm_memory_eval --brain baseline                # non-LLM baseline, no key needed
llm_memory_eval --brain ollama --model qwen3:8b # local model, no key needed
```

20 tasks in 5 categories (navigation, perception, failure handling, multi-step,
memory), defined in [`llm_memory_agent/llm_memory_agent/data/tasks.yaml`](llm_memory_agent/llm_memory_agent/data/tasks.yaml).
Memory tasks seed prior experience: objects seen, doors found closed, operator
notes, 40 distractor memories, and one deliberately stale memory. Results go to
`eval_results/summary.md` and `results.jsonl` (full tool traces).

| | memory ON | memory OFF |
|---|---|---|
| Keyword baseline (no LLM) | 30% (6/20), 7 false success claims | 30% (6/20), 7 false success claims |
| Claude Opus 5.5 | *not run yet* | *not run yet* |

### In Gazebo (Nav2 TurtleBot3)

```bash
docker compose build jazzy-sim            # once; Jazzy + Gazebo Harmonic + Nav2
docker compose run --rm jazzy-sim
colcon build && source install/setup.bash
ros2 launch llm_memory_agent agent_sim.launch.py   # or: provider:=ollama model:=<model>
# in another shell:
ros2 topic pub --once /llm_memory_agent/command std_msgs/msg/String "{data: 'Go to the north bay and tell me what you see'}"
ros2 topic echo /llm_memory_agent/response
```

Status: the Nav2 adapter and launch file are written but haven't been run in Gazebo
yet, and the named poses in `llm_memory_agent/config/tb3_sandbox.yaml` need checking
in RViz. Perception in Gazebo is simulated from configured object positions. To use
a host Ollama from the container, start it with `OLLAMA_HOST=0.0.0.0 ollama serve`;
see [TESTING.md](TESTING.md) step 6.

## Not a rosbag replacement

rosbag records *what happened* (raw, replayable sensor and state data). This memory
records *what it meant*: distilled facts at event boundaries ("navigation aborted at
lab_door_west"). Concepts point to raw logs with OKF's own `sources`/`resource` URI
fields, so they can reference rosbag2, MCAP, plain log files, or anything else
addressable, without copying them.

## Development

[TESTING.md](TESTING.md) walks through testing everything, from unit tests to the
full Gazebo stack, with fixes for common problems. The short version:

```bash
# without ROS, for the pure-Python packages:
PYTHONPATH=llm_memory_core:llm_memory_capture:llm_memory_agent pytest llm_memory_core llm_memory_capture llm_memory_agent

# with ROS, per distro (humble / jazzy / lyrical):
docker compose build jazzy      # again after changing docker/ or a package.xml
docker compose run --rm jazzy
colcon build && colcon test && colcon test-result --verbose
```

If you keep a Python virtualenv inside the repo, add a `COLCON_IGNORE` file to it
(`touch venv/COLCON_IGNORE`) so colcon doesn't try to build its contents.

Design notes, scope decisions, and the roadmap are in [requirements.md](requirements.md).
Changes per release are in [CHANGELOG.md](CHANGELOG.md); to contribute, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Credits and license

Apache-2.0. Inspired by Fabio Akita's `ai-memory` (MIT). No code is
reused, but the idea of portable, file-based LLM memory comes from there.

[okf]: https://github.com/GoogleCloudPlatform/open-knowledge-format
