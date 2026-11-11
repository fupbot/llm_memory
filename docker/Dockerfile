ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO} \
    DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-colcon-common-extensions \
      python3-rosdep \
      python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && (rosdep init || true) \
    && rosdep update --rosdistro "${ROS_DISTRO}"

# `mcp` isn't in rosdep's default keys (see okf_robotics_mcp/requirements.txt), so
# it's installed via pip rather than as a rosdep-resolved package.xml dependency.
COPY okf_robotics_mcp/requirements.txt /tmp/mcp-requirements.txt
COPY okf_robotics_agent/requirements.txt /tmp/agent-requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/mcp-requirements.txt -r /tmp/agent-requirements.txt

# Optional Gazebo + Nav2 TurtleBot3 simulation for okf_robotics_agent's
# agent_sim.launch.py (only for the `*-sim` compose services; adds a few GB).
ARG WITH_SIM=false
RUN if [ "${WITH_SIM}" = "true" ]; then \
      apt-get update && \
      if [ "${ROS_DISTRO}" = "humble" ]; then \
        apt-get install -y --no-install-recommends \
          ros-humble-nav2-bringup ros-humble-turtlebot3-gazebo; \
      else \
        apt-get install -y --no-install-recommends \
          ros-${ROS_DISTRO}-nav2-bringup ros-${ROS_DISTRO}-nav2-minimal-tb3-sim; \
      fi && rm -rf /var/lib/apt/lists/*; \
    fi

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
