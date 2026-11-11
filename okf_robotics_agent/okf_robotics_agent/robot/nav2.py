"""Nav2Robot: RobotInterface over ROS 2 Nav2, for Gazebo or real hardware.

- navigate_to -> the `NavigateToPose` action (or `NavigateThroughPoses` when `via`
  waypoints are given), with poses from a named-locations YAML.
- state -> latest pose from the localization topic (/amcl_pose by default). The
  "location" is the nearest named location within `arrival_tolerance`.
- look -> SIMULATED perception: objects listed in the config with map positions are
  "seen" when within `perception_radius`. It stands in for an object detector, so
  the agent and memory pipeline can run end to end in Gazebo. Replace it with a
  real detector topic for hardware.
- avoid_doors isn't supported: Nav2 has no named doors. A config could map doors
  to keepout-filter zones; until then the argument is reported as ignored.

Threading: blocking calls wait on futures with threading.Event, so the node must be
spun by an executor on another thread (see agent_node.py, which uses a
MultiThreadedExecutor). Calling these methods from inside a callback of a
single-threaded executor would deadlock.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from .base import NavResult, Observation, RobotInterface, RobotState


@dataclass
class NamedPose:
    name: str
    x: float
    y: float
    yaw: float = 0.0
    description: str = ""


@dataclass
class Nav2RobotConfig:
    environment_resource: str
    locations: dict[str, NamedPose]
    objects: dict[str, tuple[float, float]] = field(default_factory=dict)
    connections: list[list[str]] = field(default_factory=list)
    frame_id: str = "map"
    navigate_action: str = "navigate_to_pose"
    through_action: str = "navigate_through_poses"
    pose_topic: str = "amcl_pose"
    arrival_tolerance: float = 0.6
    perception_radius: float = 1.2
    nav_timeout_s: float = 180.0
    server_timeout_s: float = 30.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Nav2RobotConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        locations = {n: NamedPose(n, float(v["x"]), float(v["y"]), float(v.get("yaw", 0.0)),
                                  v.get("description", ""))
                     for n, v in d["locations"].items()}
        objects = {n: (float(v["x"]), float(v["y"])) for n, v in (d.get("objects") or {}).items()}
        known = {k for k in cls.__dataclass_fields__} - {"locations", "objects"}
        rest = {k: v for k, v in d.items() if k in known}
        return cls(locations=locations, objects=objects, **rest)


def _pose(frame: str, p: NamedPose, stamp) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.header.stamp = stamp
    msg.pose.position.x, msg.pose.position.y = p.x, p.y
    msg.pose.orientation.z = math.sin(p.yaw / 2.0)
    msg.pose.orientation.w = math.cos(p.yaw / 2.0)
    return msg


def _wait(future, timeout: float):
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout):
        raise TimeoutError
    return future.result()


class Nav2Robot(RobotInterface):
    def __init__(self, node: Node, config: Nav2RobotConfig):
        self.node = node
        self.cfg = config
        self._nav = ActionClient(node, NavigateToPose, config.navigate_action)
        self._through = ActionClient(node, NavigateThroughPoses, config.through_action)
        self._xy: tuple[float, float] | None = None
        self._last_xy: tuple[float, float] | None = None
        self._odometer = 0.0
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)  # amcl latches
        node.create_subscription(PoseWithCovarianceStamped, config.pose_topic, self._on_pose, qos)

    @property
    def environment_resource(self) -> str:
        return self.cfg.environment_resource

    def _on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self._last_xy is not None:
            self._odometer += math.dist(xy, self._last_xy)
        self._last_xy = self._xy = xy

    def _nearest_location(self) -> str | None:
        if self._xy is None:
            return None
        best = min(self.cfg.locations.values(), key=lambda p: math.dist((p.x, p.y), self._xy))
        return best.name if math.dist((best.x, best.y), self._xy) <= self.cfg.arrival_tolerance else None

    def state(self) -> RobotState:
        x, y = self._xy or (math.nan, math.nan)
        return RobotState(self._nearest_location(), round(x, 2), round(y, 2), round(self._odometer, 2))

    def describe_map(self) -> str:
        lines = ["Locations (map frame, meters):"]
        for p in self.cfg.locations.values():
            lines.append(f"- {p.name} ({p.x:+.2f}, {p.y:+.2f}): {p.description}")
        if self.cfg.connections:
            lines.append("Connections:")
            lines += [f"- {c[0]} <-> {c[1]}" + (f" [door: {c[2]}]" if len(c) > 2 else "")
                      for c in self.cfg.connections]
        return "\n".join(lines)

    def look(self) -> Observation:
        here = self._nearest_location()
        if self._xy is None:
            return Observation(None, [], "No pose estimate yet.")
        seen = sorted(n for n, xy in self.cfg.objects.items()
                      if math.dist(xy, self._xy) <= self.cfg.perception_radius)
        desc = self.cfg.locations[here].description if here else "Between named locations."
        return Observation(here, seen, desc)

    def navigate_to(self, location, *, via=None, avoid_doors=None) -> NavResult:
        names = [*(via or []), location]
        unknown = [n for n in names if n not in self.cfg.locations]
        if unknown:
            return NavResult(False, self._nearest_location(),
                             f"Unknown location(s) {unknown}. Known: {', '.join(self.cfg.locations)}.")
        note = (f" (avoid_doors={list(avoid_doors)} ignored: not supported with Nav2.)"
                if avoid_doors else "")
        start_odo = self._odometer
        stamp = self.node.get_clock().now().to_msg()
        poses = [_pose(self.cfg.frame_id, self.cfg.locations[n], stamp) for n in names]
        client = self._through if via else self._nav
        goal = NavigateThroughPoses.Goal() if via else NavigateToPose.Goal()
        if via:
            goal.poses = poses
        else:
            goal.pose = poses[0]

        if not client.wait_for_server(timeout_sec=self.cfg.server_timeout_s):
            return NavResult(False, self._nearest_location(),
                             "Nav2 action server not available. Is Nav2 running?" + note)
        try:
            handle = _wait(client.send_goal_async(goal), self.cfg.server_timeout_s)
            if not handle.accepted:
                return NavResult(False, self._nearest_location(), "Nav2 rejected the goal." + note)
            wrapped = _wait(handle.get_result_async(), self.cfg.nav_timeout_s)
        except TimeoutError:
            return NavResult(False, self._nearest_location(),
                             f"Navigation timed out after {self.cfg.nav_timeout_s:.0f}s." + note)

        here = self._nearest_location()
        travelled = round(self._odometer - start_odo, 2)
        if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
            return NavResult(True, here or location, f"Arrived at {location}." + note,
                             names, travelled)
        # error_code/error_msg exist on Jazzy+ results; Humble's result is empty.
        code = getattr(wrapped.result, "error_code", None)
        detail = getattr(wrapped.result, "error_msg", "") or ""
        status = {GoalStatus.STATUS_ABORTED: "aborted", GoalStatus.STATUS_CANCELED: "canceled"}.get(
            wrapped.status, f"status {wrapped.status}")
        msg = f"Navigation {status}" + (f" (error {code}: {detail})" if code else "")
        msg += f". Robot is at {here or 'an unnamed spot'}." + note
        return NavResult(False, here, msg, [], travelled)
