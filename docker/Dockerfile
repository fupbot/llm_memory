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

# `mcp` isn't in rosdep's default keys (see llm_memory_mcp/requirements.txt), so
# it's installed via pip rather than as a rosdep-resolved package.xml dependency.
COPY llm_memory_mcp/requirements.txt /tmp/mcp-requirements.txt
COPY llm_memory_agent/requirements.txt /tmp/agent-requirements.txt
# Some dependencies want newer versions of packages that apt installed (PyYAML,
# typing-extensions, ...), and pip can't uninstall those ("installed by debian").
# --ignore-installed puts pip's copies in /usr/local alongside them instead. numpy
# stays on 1.x, which ROS's own Python packages are built against.
RUN echo "numpy<2" > /tmp/constraints.txt \
    && pip3 install --no-cache-dir --ignore-installed -c /tmp/constraints.txt \
       -r /tmp/mcp-requirements.txt -r /tmp/agent-requirements.txt \
    && rm /tmp/constraints.txt

# Optional Gazebo + Nav2 TurtleBot3 simulation for llm_memory_agent's
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

# Install the packages' rosdep dependencies at build time, so containers start with
# them in place. Only the package.xml files are copied, so editing source code
# doesn't invalidate this layer. entrypoint.sh still installs anything added later.
COPY llm_memory_core/package.xml /tmp/deps/llm_memory_core/package.xml
COPY llm_memory_capture/package.xml /tmp/deps/llm_memory_capture/package.xml
COPY llm_memory_mcp/package.xml /tmp/deps/llm_memory_mcp/package.xml
COPY llm_memory_agent/package.xml /tmp/deps/llm_memory_agent/package.xml
RUN apt-get update \
    && rosdep install --from-paths /tmp/deps --ignore-src -r -y \
    && rm -rf /var/lib/apt/lists/* /tmp/deps

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
