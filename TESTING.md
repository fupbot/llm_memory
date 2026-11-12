# Testing robot_ai_memory

Steps to test the whole project, from the fastest check to the full Gazebo stack.
Each step builds on the previous one. Steps 0–3 need no ROS; steps 4–6 use Docker.

`MODEL` below is the LLM you test with: an Ollama model you already have (see
`ollama list`), or leave it out and use the Anthropic default.

## 0. Set up a Python environment (once)

```bash
cd robot_ai_memory
python3 -m venv venv && source venv/bin/activate
pip install -e llm_memory_core -e llm_memory_agent -e llm_memory_mcp mcp pytest
touch venv/COLCON_IGNORE   # keep colcon (step 4) out of the virtualenv; see Troubleshooting
```

Pick an LLM provider (details in the README's "Choosing the LLM"):

```bash
# Local, with Ollama (no key):
export MODEL=your-model:tag
PROVIDER="--provider ollama --model $MODEL"
# or Claude:
export ANTHROPIC_API_KEY=...
PROVIDER=""
```

If Ollama doesn't run on `localhost:11434` (for example, it's on the Windows side of
WSL2), set `OLLAMA_HOST=<host>:11434`. For local models, start Ollama with a larger
context window, e.g. `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`.

## 1. Unit tests (no ROS, no API key)

```bash
PYTHONPATH=llm_memory_core:llm_memory_capture:llm_memory_agent \
  pytest llm_memory_core llm_memory_capture llm_memory_agent
```

## 2. Agent memory across sessions (no ROS)

```bash
rm -rf ~/llm_memory_bundle   # start with empty memory
llm_memory_agent_sim $PROVIDER -c "Go to the lab." --closed-door lab_door_west
llm_memory_agent_sim $PROVIDER -c "Go to the lab." --closed-door lab_door_west
find ~/llm_memory_bundle -name '*.md' | head
```

Expected: in run 1 the robot hits `lab_door_west`, reroutes, and the failure is
written to memory. In run 2 the first `navigate_to` already passes
`avoid_doors: ["lab_door_west"]`.

Then the evaluation, memory ON vs OFF:

```bash
llm_memory_eval --brain baseline --out eval_results/baseline   # reference: 30%
llm_memory_eval --brain ollama --model $MODEL --memory both --out eval_results/local
# or Claude: llm_memory_eval --memory both --out eval_results/claude
cat eval_results/local/summary.md
```

This is 40 episodes, run one at a time on Ollama. Add `--tasks <id> ...` (ids from
`llm_memory_agent/llm_memory_agent/data/tasks.yaml`) for a quick subset.
Expected: memory ON beats OFF on the memory tasks, with few false success claims.

## 3. MCP server: query the memory from an LLM client

```bash
LLM_MEMORY_BUNDLE_ROOT=~/llm_memory_bundle llm_memory_mcp
```

This only checks that it starts: an MCP server talks over stdin/stdout, so it prints
nothing and waits. Stop it with Ctrl-C. To really test it, register it with an MCP
client, e.g. Claude Code:

```bash
claude mcp add llm-memory -e LLM_MEMORY_BUNDLE_ROOT=$HOME/llm_memory_bundle \
  -- $PWD/venv/bin/llm_memory_mcp
```

Then, in a new session, ask "use llm-memory: what happened at lab_door_west?". It
should find the failure from step 2.

## 4. ROS build and tests, per distro (Docker)

Build the image first, and again after changing `docker/` or any `package.xml`:
`docker compose run` reuses an existing image and won't pick up those changes.

```bash
docker compose build jazzy
docker compose run --rm jazzy
# inside the container:
rm -rf build/* install/* log/*   # clean build (the build dirs persist in Docker volumes)
colcon build && colcon test && colcon test-result --verbose
```

Expected: 4 packages (`llm_memory_core`, `_capture`, `_mcp`, `_agent`) and no
failures. Repeat with `humble` and `lyrical`.

## 5. Capture node (Docker)

In the `jazzy` container, after step 4:

```bash
source install/setup.bash
ros2 launch llm_memory_capture capture.launch.py &
ros2 topic pub --once /diagnostics diagnostic_msgs/msg/DiagnosticArray \
  "{status: [{level: 2, name: 'motor', message: 'overcurrent'}]}"
find ~/llm_memory_bundle -name '*.md' -newer /proc/1   # a new failure note
```

The node also listens on `/navigate_to_pose/_action/status` and `/plan`; those are
exercised by the full stack in step 6.

## 6. Full stack: Gazebo + Nav2 + agent + capture (Docker)

Status: not yet run end to end. Expect to adjust things, especially the named poses
in `llm_memory_agent/config/tb3_sandbox.yaml` (check them in RViz).

If the LLM is Ollama on the host, it must listen on all interfaces so the container
can reach it (by default it only accepts connections from the host itself):

```bash
OLLAMA_HOST=0.0.0.0 OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

The `jazzy-sim` container reaches it through `host.docker.internal`. If Ollama runs
elsewhere (e.g. on Windows under WSL2), export `OLLAMA_HOST=http://<ip>:11434`
before `docker compose run`.

```bash
docker compose build jazzy-sim   # once; several GB (Gazebo + Nav2)
docker compose run --rm jazzy-sim
# inside the container:
colcon build && source install/setup.bash
ros2 launch llm_memory_agent agent_sim.launch.py provider:=ollama model:=<your-model> &
#   (Claude: ros2 launch llm_memory_agent agent_sim.launch.py)
#   add headless:=False to watch Gazebo
ros2 launch llm_memory_capture capture.launch.py &
ros2 topic echo /llm_memory_agent/response &
ros2 topic pub --once /llm_memory_agent/command std_msgs/msg/String \
  "{data: 'Go to the north bay and tell me what you see'}"
```

Expected: the node logs `Ready. LLM ollama:<model>`, Nav2 drives the TurtleBot, a
JSON response is published, and both the agent and the capture node write notes to
`/root/llm_memory_bundle` (a Docker volume).

## Troubleshooting

**`docker compose` needs `sudo`, and the container doesn't see `ANTHROPIC_API_KEY` /
`OLLAMA_HOST`.** `sudo` drops your environment variables. Use `sudo -E docker compose
...`, or add yourself to the docker group so `sudo` isn't needed:
`sudo usermod -aG docker $USER`, then open a new terminal (on WSL2: `wsl --shutdown`).

**`colcon build` errors about `venv/lib/.../numpy/...`, Cython, or a package called
`client` failing with "NO TESTS RAN".** colcon scans every folder of the mounted
repo, including a Python virtualenv, and tries to build the `setup.py` files inside
it. Run `touch venv/COLCON_IGNORE` (the container's entrypoint also does this for
virtualenvs in the repo), then rerun the clean build from step 4.

**`rosdep` fails with `Unable to locate package ros-<distro>-nav2-...` when the
container starts.** The image is older than a `package.xml` change. Rebuild it:
`docker compose build <distro>`. (Dependencies are installed at image build time;
the entrypoint only installs ones added since, and only warns if that fails.)

**Image build fails in `pip3 install` with `Cannot uninstall PyYAML ... installed by
debian`.** Fixed in `docker/Dockerfile` (pip installs with `--ignore-installed`,
numpy pinned below 2 for ROS). If you see it, your Dockerfile predates the fix.

**`colcon test` fails every package with `Ran 0 tests ... NO TESTS RAN` (seen on
Lyrical), with `Unknown distribution option: 'tests_require'` warnings.** Newer
setuptools dropped `tests_require`, so colcon didn't know to use pytest and ran
`unittest`, which doesn't find pytest-style tests. Fixed: each `setup.py` declares
`extras_require={"test": ["pytest"]}`. If you see it, your `setup.py` predates the
fix. (`llm_memory_mcp` has no tests yet; pytest's "no tests collected" is not a
failure in colcon.)

**Old `okf_robotics_*` names still show up (in colcon output, your venv, or memory).**
The packages were renamed to `llm_memory_*` before 0.1.0. Do the clean build from
step 4 (`rm -rf build/* install/* log/*` in each container), reinstall in your
virtualenv (`pip uninstall okf_robotics_core okf_robotics_agent okf_robotics_mcp`,
then step 0), and move existing memory: `mv ~/okf_robotics_bundle ~/llm_memory_bundle`.

**`llm_memory_mcp: command not found`.** The MCP package isn't installed in the
virtualenv: `pip install -e llm_memory_mcp mcp`.

**The agent gets `can't reach http://.../v1 ... is the server running?`.** Ollama
isn't running at that address. Check `curl $OLLAMA_HOST/api/tags` (or
`localhost:11434`), and inside Docker see step 6.

**A local model loops, forgets to call `finish`, or makes malformed tool calls.** Use
a model trained for tool calling (Qwen3, Llama 3.1+, Mistral Small), a larger one if
you can, and a bigger Ollama context window (`OLLAMA_CONTEXT_LENGTH`).
