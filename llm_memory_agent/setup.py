from glob import glob

from setuptools import find_packages, setup

package_name = "llm_memory_agent"

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
    url="https://github.com/fupbot/robot_ai_memory",
    python_requires=">=3.10",
    extras_require={"test": ["pytest"]},  # tells colcon to run tests with pytest
    entry_points={
        "console_scripts": [
            "agent_node = llm_memory_agent.agent_node:main",
            "llm_memory_agent_sim = llm_memory_agent.cli:main",
            "llm_memory_eval = llm_memory_agent.eval.runner:main",
        ],
    },
)
