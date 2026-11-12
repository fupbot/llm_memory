# Contributing

Thanks for helping. Bug reports, fixes and new capture triggers or concept types
are all welcome.

## Reporting a problem

Open an issue at <https://github.com/fupbot/robot_ai_memory/issues> with your ROS
distro, how you installed (Docker, source, pip), the command you ran, and the full
error. Check the Troubleshooting section of [TESTING.md](TESTING.md) first: it covers
the common setup problems.

## Making a change

1. Fork the repo and branch from `dev`.
2. Set up and run the tests as in [TESTING.md](TESTING.md):
   - pure-Python tests (fast, no ROS): step 1;
   - ROS build and tests per distro in Docker: step 4.
3. Add tests for what you change. Code that needs a model or a robot should take a
   fake (see `ScriptedBrain` and the fake-server tests in `llm_memory_agent/test/`),
   so tests run offline and without API keys.
4. Update the docs in the same change: README.md for user-facing behavior,
   TESTING.md for new test steps or troubleshooting, and the package's
   `CHANGELOG.rst` under a "Forthcoming" heading.
5. Open a pull request against `dev`. CI runs the Python tests and a colcon build
   and test on Humble, Jazzy and Lyrical; all of them must pass.

Design decisions and the roadmap are in [requirements.md](requirements.md). For a
larger change, open an issue first to discuss it.

## Code conventions

- Python 3.10+ (Humble's floor). `llm_memory_core` and `llm_memory_mcp` must not
  import rclpy, so they stay usable outside ROS.
- Keep ROS-specific code at the edges (nodes, adapters); logic should be testable
  without ROS.
- Never take API keys as command-line arguments or ROS parameters: read them from the
  environment.

## Releasing (maintainers)

1. On `dev`, make sure CI is green on all three distros.
2. Bump the version in every package's `package.xml` and `setup.py`, and in the
   `generated_by` strings (`llm_memory_capture/llm_memory_capture/encoders/*.py`,
   `llm_memory_agent/llm_memory_agent/memory.py`). All packages share one version.
3. In each `CHANGELOG.rst`, rename "Forthcoming" to `X.Y.Z (YYYY-MM-DD)`, and add the
   release to the top-level [CHANGELOG.md](CHANGELOG.md).
4. Merge `dev` into `main`, tag `vX.Y.Z` on `main`, push the tag, and create a GitHub
   release from it using the CHANGELOG.md entry.

By contributing, you agree that your contributions are licensed under the
Apache-2.0 license (see [LICENSE](LICENSE)).
