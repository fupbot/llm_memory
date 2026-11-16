from setuptools import find_packages, setup

package_name = "llm_memory_mcp"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "mcp"],
    zip_safe=True,
    maintainer="Ugalde Pereira",
    maintainer_email="ugaldepereira@gmail.com",
    description="MCP server exposing read-only query/search over an OKF robotics bundle.",
    license="Apache-2.0",
    url="https://github.com/fupbot/robot_ai_memory",
    python_requires=">=3.10",
    extras_require={"test": ["pytest"]},  # tells colcon to run tests with pytest
    entry_points={
        "console_scripts": [
            "llm_memory_mcp = llm_memory_mcp.server:main",
        ],
    },
)
