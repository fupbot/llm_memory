^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package llm_memory_core
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.0 (2026-11-11)
------------------
* First release.
* Read and write OKF (Open Knowledge Format) bundles: markdown files with YAML
  frontmatter, plus links between concepts.
* Robotics concept types (Mission Run, Task Attempt, Replan Event,
  Failure/Recovery Note, Environment Observation, Param/Planner Change) and a bundle
  validator.
* Search backends behind one interface: grep (exact, no dependencies) and vector
  search (Chroma, with MiniLM or hashing embeddings).
* Pure Python, no rclpy: usable outside ROS.
