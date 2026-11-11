"""okf_robotics_agent_sim: talk to the agent in the built-in simulator, no ROS needed.

    okf_robotics_agent_sim                                 # interactive
    okf_robotics_agent_sim -c "Go to the lab." --closed-door lab_door_west
    okf_robotics_agent_sim --memory off                    # ablation

Each run of this command is one session. Memory persists across runs in --bundle,
so a door found closed in one run is retrieved in the next.
"""
from __future__ import annotations

import argparse
import sys

from .agent import RobotAgent, format_result
from .brain import AnthropicBrain
from .memory import open_memory
from .robot.sim import SimRobot
from .world import World


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--command", action="append", help="run these commands then exit")
    p.add_argument("--world", help="world YAML (default: built-in house)")
    p.add_argument("--closed-door", action="append", default=[], help="door closed in this run")
    p.add_argument("--bundle", default="~/okf_robotics_bundle", help="OKF bundle (long-term memory)")
    p.add_argument("--memory", choices=["on", "off"], default="on")
    p.add_argument("--embedder", choices=["minilm", "hashing"], default="minilm")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    args = p.parse_args(argv)

    world = World.from_yaml(args.world) if args.world else World.builtin()
    world = world.with_overrides(closed_doors=args.closed_door)
    robot = SimRobot(world)
    memory = None
    if args.memory == "on":
        memory = open_memory(args.bundle, world.resource, embedder=args.embedder)
        memory.start_session("sim")
    agent = RobotAgent(robot, AnthropicBrain(args.model, effort=args.effort), memory)

    def run(cmd: str) -> None:
        print(format_result(agent.run(cmd)), flush=True)

    try:
        if args.command:
            for cmd in args.command:
                run(cmd)
        else:
            print(f"Robot at {robot.state().location} in {world.resource}. "
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
