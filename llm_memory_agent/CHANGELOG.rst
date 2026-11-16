^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package llm_memory_agent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.0 (2026-11-11)
------------------
* First release.
* LLM agent that turns natural-language commands into robot actions (navigate to
  named places, look, search memory, remember, finish) and reports success honestly.
* Retrieval-augmented long-term memory: each episode is written to an OKF bundle and
  retrieved in later sessions through a vector index.
* LLM providers: Anthropic (Claude), local Ollama models, or any OpenAI-compatible
  API (``--provider`` / ``provider:=``).
* Built-in simulator (``llm_memory_agent_sim``) and an evaluation set of 20 tasks
  (``llm_memory_eval``), memory on vs off, graded against the simulator's ground
  truth. Neither needs ROS.
* ROS 2 node with a Nav2 adapter and a Gazebo TurtleBot3 launch file
  (``agent_sim.launch.py``; experimental, not yet run end to end).
