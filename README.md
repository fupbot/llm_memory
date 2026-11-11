# robot_ai_memory

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
| `okf_robotics_core` | OKF bundle read/write, validation, grep + vector search | no | no |
| `okf_robotics_capture` | ROS 2 node that writes task/replan/failure events to the bundle, non-blocking | yes | no |
| `okf_robotics_mcp` | MCP server: query the bundle from any LLM client | no | no |
| `okf_robotics_agent` | Claude-driven agent: natural language → Nav2, RAG memory, simulator, eval | optional | yes |

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
pip install -e okf_robotics_core -e okf_robotics_agent   # or: pip install pyyaml anthropic chromadb
export ANTHROPIC_API_KEY=...

# Session 1: the west lab door is closed. The agent should hit it, reroute, and the
# failure is written to memory.
okf_robotics_agent_sim -c "Go to the lab." --closed-door lab_door_west
# Session 2: the failure is retrieved, so it can plan around the door from the start.
okf_robotics_agent_sim -c "Go to the lab." --closed-door lab_door_west
# Memory lives here, as markdown:
find ~/okf_robotics_bundle -name '*.md' | head
```

### Evaluation

```bash
okf_robotics_eval --memory both --trials 3        # Claude, memory ON vs OFF
okf_robotics_eval --brain baseline                # non-LLM baseline, no key needed
```

20 tasks in 5 categories (navigation, perception, failure handling, multi-step,
memory), defined in [`okf_robotics_agent/okf_robotics_agent/data/tasks.yaml`](okf_robotics_agent/okf_robotics_agent/data/tasks.yaml).
Memory tasks seed prior experience: objects seen, doors found closed, operator
notes, 40 distractor memories, and one deliberately stale memory. Results go to
`eval_results/summary.md` and `results.jsonl` (full tool traces).

| | memory ON | memory OFF |
|---|---|---|
| Keyword baseline (no LLM) | 30% (6/20), 7 false success claims | 30% (6/20), 7 false success claims |
| Claude Opus 5.5 | *not run yet* | *not run yet* |

### In Gazebo (Nav2 TurtleBot3)

```bash
docker compose run --rm jazzy-sim          # Jazzy + Gazebo Harmonic + Nav2
colcon build && source install/setup.bash
ros2 launch okf_robotics_agent agent_sim.launch.py
# in another shell:
ros2 topic pub --once /okf_robotics_agent/command std_msgs/msg/String "{data: 'Go to the north bay and tell me what you see'}"
ros2 topic echo /okf_robotics_agent/response
```

Status: the Nav2 adapter and launch file are written but haven't been run in Gazebo
yet, and the named poses in `okf_robotics_agent/config/tb3_sandbox.yaml` need checking
in RViz. Perception in Gazebo is simulated from configured object positions.

## Not a rosbag replacement

rosbag records *what happened* (raw, replayable sensor and state data). This memory
records *what it meant*: distilled facts at event boundaries ("navigation aborted at
lab_door_west"). Concepts point to raw logs with OKF's own `sources`/`resource` URI
fields, so they can reference rosbag2, MCAP, plain log files, or anything else
addressable, without copying them.

## Development

```bash
docker compose run --rm jazzy   # or humble / lyrical
colcon build && colcon test
# or, without ROS, for the pure-Python packages:
PYTHONPATH=okf_robotics_core:okf_robotics_capture:okf_robotics_agent pytest okf_robotics_core okf_robotics_capture okf_robotics_agent
```

Design notes, scope decisions, and the roadmap are in [requirements.md](requirements.md).

## Credits and license

Apache-2.0. Inspired by Fabio Akita's `ai-memory` (MIT). No code is
reused, but the idea of portable, file-based LLM memory comes from there.

[okf]: https://github.com/GoogleCloudPlatform/open-knowledge-format
