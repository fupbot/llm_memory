import pytest

from okf_robotics_agent.robot.sim import SimRobot
from okf_robotics_agent.world import World


@pytest.fixture
def world():
    return World.builtin("house")


def test_builtin_world_loads_and_plans(world):
    assert world.start == "dock"
    assert world.shortest_path("dock", "lab") == ["dock", "hallway", "living_room", "lab"]
    assert world.shortest_path("dock", "lab", excluded_doors={"lab_door_west"}) == [
        "dock", "hallway", "kitchen", "dining", "lab"]
    assert world.shortest_path("dock", "garage", excluded_doors={"garage_door"}) is None


def test_overrides_and_validation(world):
    w = world.with_overrides(closed_doors=["garage_door"], objects={"toolbox": "office", "keys": None})
    assert w.objects["toolbox"] == "office" and "keys" not in w.objects
    assert world.objects["toolbox"] == "garage"  # original untouched
    with pytest.raises(ValueError):
        world.with_overrides(closed_doors=["no_such_door"])
    with pytest.raises(ValueError):
        world.with_overrides(objects={"toolbox": "attic"})


def test_navigate_success_updates_state(world):
    robot = SimRobot(world)
    res = robot.navigate_to("kitchen")
    assert res.success and res.location == "kitchen"
    assert res.path == ["hallway", "kitchen"]
    assert robot.state().odometer == pytest.approx(5.0)
    assert robot.look().objects == ["fridge", "red_mug"]


def test_closed_door_aborts_then_session_costmap_avoids_it(world):
    robot = SimRobot(world.with_overrides(closed_doors=["lab_door_west"]))
    first = robot.navigate_to("lab")
    assert not first.success and first.blocked_door == "lab_door_west"
    assert first.location == "living_room"  # stopped on the near side of the door
    retry = robot.navigate_to("lab")
    assert retry.success and retry.path == ["dining", "lab"]
    # A new session forgets what was learned, like a fresh costmap.
    robot.reset_session()
    assert robot.session_blocked_doors == set()


def test_avoid_doors_plans_around_without_hitting_door(world):
    robot = SimRobot(world.with_overrides(closed_doors=["lab_door_west"]))
    res = robot.navigate_to("lab", avoid_doors=["lab_door_west"])
    assert res.success and "living_room" not in res.path


def test_via_waypoints_and_errors(world):
    robot = SimRobot(world)
    res = robot.navigate_to("dock", via=["garage"])
    assert res.success and res.path == ["garage", "dock"]
    assert not robot.navigate_to("roof").success
    assert not robot.navigate_to("lab", avoid_doors=["front_door"]).success
    blocked = SimRobot(world.with_overrides(closed_doors=["garage_door"]))
    assert not blocked.navigate_to("garage", avoid_doors=["garage_door"]).success
    assert blocked.trajectory == ["dock"]
