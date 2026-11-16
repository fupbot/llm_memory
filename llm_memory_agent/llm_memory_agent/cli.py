"""llm_memory_agent_sim: talk to the agent in the built-in simulator, no ROS needed.

    llm_memory_agent_sim                                 # interactive
    llm_memory_agent_sim -c "Go to the lab." --closed-door lab_door_west
    llm_memory_agent_sim --memory off                    # ablation
    llm_memory_agent_sim --provider ollama --model qwen3:8b      # local LLM
    llm_memory_agent_sim --provider openai --model gpt-5-mini     # OpenAI-compatible cloud

API keys come from the environment: ANTHROPIC_API_KEY for --provider anthropic,
OPENAI_API_KEY for --provider openai. Ollama needs no key.

Each run of this command is one session. Memory persists across runs in --bundle,
so a door found closed in one run is retrieved in the next.
"""
from __future__ import annotations

import argparse
import sys

from .agent import RobotAgent, format_result
from .brain import PROVIDERS, make_brain
from .memory import open_memory
from .robot.sim import SimRobot
from .world import World


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--command", action="append", help="run these commands then exit")
    p.add_argument("--world", help="world YAML (default: built-in house)")
    p.add_argument("--closed-door", action="append", default=[], help="door closed in this run")
    p.add_argument("--bundle", default="~/llm_memory_bundle", help="OKF bundle (long-term memory)")
    p.add_argument("--memory", choices=["on", "off"], default="on")
    p.add_argument("--embedder", choices=["minilm", "hashing"], default="minilm")
    p.add_argument("--provider", choices=list(PROVIDERS), default="anthropic",
                   help="LLM provider (default: anthropic)")
    p.add_argument("--model", help="model name (default: claude-opus-5-5 for anthropic, "
                   "qwen3:8b for ollama; required for openai)")
    p.add_argument("--base-url", help="API base URL for ollama/openai "
                   "(default: $OLLAMA_HOST or localhost:11434 for ollama, $OPENAI_BASE_URL "
                   "or api.openai.com for openai)")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
                   help="Claude effort level (anthropic only)")
    args = p.parse_args(argv)

    try:
        brain = make_brain(args.provider, args.model, effort=args.effort, base_url=args.base_url)
    except ValueError as exc:
        p.error(str(exc))

    world = World.from_yaml(args.world) if args.world else World.builtin()
    world = world.with_overrides(closed_doors=args.closed_door)
    robot = SimRobot(world)
    memory = None
    if args.memory == "on":
        memory = open_memory(args.bundle, world.resource, embedder=args.embedder)
        memory.start_session("sim")
    agent = RobotAgent(robot, brain, memory)

    def run(cmd: str) -> None:
        print(format_result(agent.run(cmd)), flush=True)

    try:
        if args.command:
            for cmd in args.command:
                run(cmd)
        else:
            print(f"Robot at {robot.state().location} in {world.resource}, using {brain.name}. "
                  "Type a command (Ctrl-D to quit).")
            for line in sys.stdin:
                if line.strip():
                    run(line.strip())
    finally:
        if memory is not None:
            memory.end_session()
    return 0


if __name__ == "__main__":
    sys.exit(main())
