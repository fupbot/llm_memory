"""llm_memory_agent's ROS 2 node: natural-language commands in, robot actions out.

    ros2 topic pub --once /llm_memory_agent/command std_msgs/msg/String "{data: 'Go to the north bay'}"
    ros2 topic echo /llm_memory_agent/response

LLM: parameter `provider` is "anthropic" (ANTHROPIC_API_KEY), "ollama" (local, base
URL from `base_url`, else OLLAMA_HOST, else localhost:11434) or "openai" (any
OpenAI-compatible API, OPENAI_API_KEY). An empty `model` uses the provider's default.

Commands are queued and run one at a time on a worker thread, so a slow LLM call
never blocks the executor. The executor (multi-threaded) keeps serving the Nav2
action client and pose subscription while the agent waits on them.
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from .agent import RobotAgent, format_result
from .brain import make_brain
from .memory import open_memory
from .robot.nav2 import Nav2Robot, Nav2RobotConfig


class AgentNode(Node):
    def __init__(self):
        super().__init__("llm_memory_agent")
        default_cfg = str(Path(get_package_share_directory("llm_memory_agent"))
                          / "config" / "tb3_sandbox.yaml")
        p = {
            "robot_config": default_cfg,
            "bundle_root": "~/llm_memory_bundle",
            "index_dir": "",
            "memory_enabled": True,
            "embedder": "minilm",
            "provider": "anthropic",
            "model": "",
            "base_url": "",
            "effort": "medium",
            "max_steps": 20,
        }
        for name, default in p.items():
            self.declare_parameter(name, default)
        get = lambda n: self.get_parameter(n).value  # noqa: E731

        self.robot = Nav2Robot(self, Nav2RobotConfig.from_yaml(get("robot_config")))
        self.memory = None
        if get("memory_enabled"):
            self.memory = open_memory(get("bundle_root"), self.robot.environment_resource,
                                      index_dir=get("index_dir") or None, embedder=get("embedder"))
            self.memory.start_session("ros")
        brain = make_brain(get("provider"), get("model") or None, effort=get("effort"),
                           base_url=get("base_url") or None)
        self.agent = RobotAgent(self.robot, brain, self.memory, max_steps=int(get("max_steps")))

        self._pub = self.create_publisher(String, "~/response", 10)
        self.create_subscription(String, "~/command", self._on_command, 10)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()
        self.get_logger().info(
            f"Ready. LLM {brain.name}, memory {'ON' if self.memory else 'OFF'}; send commands on "
            f"{self.resolve_topic_name('~/command')}")

    def _on_command(self, msg: String) -> None:
        self.get_logger().info(f"Command queued: {msg.data!r}")
        self._queue.put(msg.data)

    def _work(self) -> None:
        while (cmd := self._queue.get()) is not None:
            try:
                result = self.agent.run(cmd)
                self.get_logger().info("\n" + format_result(result))
                payload = {k: v for k, v in result.to_dict().items() if k != "trace"}
                self._pub.publish(String(data=json.dumps(payload)))
            except Exception as exc:  # keep the node alive for the next command
                self.get_logger().error(f"Agent failed on {cmd!r}: {exc}")

    def shutdown(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=5)
        if self.memory is not None:
            self.memory.end_session()


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
