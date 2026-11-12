# Changelog

All notable changes to this repository. Each package also has its own
`CHANGELOG.rst` (the format ROS release tooling reads):
[llm_memory_core](llm_memory_core/CHANGELOG.rst),
[llm_memory_capture](llm_memory_capture/CHANGELOG.rst),
[llm_memory_mcp](llm_memory_mcp/CHANGELOG.rst),
[llm_memory_agent](llm_memory_agent/CHANGELOG.rst).

## 0.1.0 (2026-11-11)

First release. Supports ROS 2 Humble, Jazzy and Lyrical; Python 3.10+.

- **llm_memory_core**: read, write, validate and search OKF (Open Knowledge Format)
  bundles of robot memory. Grep and vector (Chroma) search. Pure Python, no ROS.
- **llm_memory_capture**: ROS 2 node that records task, replan and diagnostics
  events to the bundle through a non-blocking writer. Topics are configurable.
- **llm_memory_mcp**: MCP server so any MCP-capable LLM client can query the bundle.
- **llm_memory_agent**: LLM agent that drives a robot from natural-language commands
  with retrieval-augmented memory. Works with Claude, local Ollama models, or any
  OpenAI-compatible API. Includes a simulator and a 20-task evaluation set (no ROS
  needed) and a Nav2 adapter.

Known limitations:

- The Gazebo + Nav2 demo (`agent_sim.launch.py`) hasn't been run end to end yet;
  treat it as experimental.
- `llm_memory_mcp` has no automated tests yet, and `llm_memory_capture` has
  regression tests only.
- Not yet released as ROS binary packages (apt): build from source. The agent's
  `anthropic` and `chromadb` dependencies, and the MCP server's `mcp`, come from PyPI.
