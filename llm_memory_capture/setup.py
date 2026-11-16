import os
from glob import glob

from setuptools import find_packages, setup

package_name = "llm_memory_capture"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ugalde Pereira",
    maintainer_email="ugaldepereira@gmail.com",
    description=(
        "ROS2 node that captures task/planning/diagnostic events into an OKF bundle "
        "via a non-blocking buffered writer."
    ),
    license="Apache-2.0",
    url="https://github.com/fupbot/robot_ai_memory",
    python_requires=">=3.10",
    extras_require={"test": ["pytest"]},  # tells colcon to run tests with pytest
    entry_points={
        "console_scripts": [
            "capture_node = llm_memory_capture.capture_node:main",
        ],
    },
)
