from glob import glob

from setuptools import find_packages, setup

package_name = "okf_robotics_agent"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    package_data={package_name: ["data/*.yaml"]},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "PyYAML", "anthropic", "chromadb"],
    zip_safe=False,
    maintainer="Ugalde Pereira",
    maintainer_email="ugaldepereira@gmail.com",
    description=(
        "LLM agent that controls a ROS 2 robot from natural-language commands, with "
        "retrieval-augmented long-term memory in an OKF bundle."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "agent_node = okf_robotics_agent.agent_node:main",
            "okf_robotics_agent_sim = okf_robotics_agent.cli:main",
            "okf_robotics_eval = okf_robotics_agent.eval.runner:main",
        ],
    },
)
