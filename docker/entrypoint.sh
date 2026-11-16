#!/bin/bash
set -e

source "/opt/ros/${ROS_DISTRO}/setup.bash"

# The bind-mounted repo may contain a host Python virtualenv (venv/, .venv/). Its
# site-packages hold setup.py files that colcon would try to build as packages.
if [ -d /workspace/src ]; then
  find /workspace/src -maxdepth 3 -name pyvenv.cfg -printf '%h\n' 2>/dev/null \
    | while read -r venv_dir; do touch "$venv_dir/COLCON_IGNORE" 2>/dev/null || true; done
fi

# The repo is bind-mounted at /workspace/src/robot_ai_memory, so dependencies
# declared in package.xml are (re-)resolved on every start to pick up edits
# made on the host since the image was built. The image has no apt package lists
# (removed to save space), so refresh them only when something is missing, and
# don't abort the container if a dependency can't be installed: warn instead.
if [ -d /workspace/src ] && ! rosdep check --from-paths /workspace/src --ignore-src >/dev/null 2>&1; then
  echo "Installing missing rosdep dependencies..."
  apt-get update -qq || true
  rosdep install --from-paths /workspace/src --ignore-src -r -y \
    || echo "WARNING: some rosdep dependencies failed to install; see above." >&2
fi

if [ -f /workspace/install/setup.bash ]; then
  source /workspace/install/setup.bash
fi

exec "$@"
