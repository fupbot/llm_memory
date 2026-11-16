^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package llm_memory_mcp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.0 (2026-11-11)
------------------
* First release.
* MCP server (``llm_memory_mcp``) with read-only tools over an OKF bundle:
  ``search``, ``get_concept``, ``graph_neighbors``, ``environment_history``,
  ``recent``.
* Bundle location from the ``LLM_MEMORY_BUNDLE_ROOT`` environment variable.
* Works with both the 1.x and 2.x ``mcp`` Python SDK.
