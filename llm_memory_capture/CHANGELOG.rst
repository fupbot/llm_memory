^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package llm_memory_capture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.0 (2026-11-11)
------------------
* First release.
* ROS 2 node that writes OKF concepts to a bundle on meaningful events: task
  lifecycle (Nav2 action status), replans, and diagnostics errors.
* Non-blocking buffered writer, so disk I/O never blocks the ROS executor.
* Topics and message types come from config (``config/capture_params.yaml``), not
  hardcoded to one Nav2 version.
* ``capture.launch.py`` launch file.
