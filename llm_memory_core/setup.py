from setuptools import find_packages, setup

package_name = "llm_memory_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="Ugalde Pereira",
    maintainer_email="ugaldepereira@gmail.com",
    description=(
        "Pure-Python OKF bundle read/write, schema validation, and search library "
        "for robotics-specific memory concepts."
    ),
    license="Apache-2.0",
    url="https://github.com/fupbot/robot_ai_memory",
    python_requires=">=3.10",
    extras_require={"test": ["pytest"]},  # tells colcon to run tests with pytest
)
