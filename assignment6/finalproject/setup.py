from setuptools import setup, find_packages

setup(
    name="final_project",
    version="0.1.0",
    author="Morten Blørstad",
    author_email="your.email@example.com",
    description="A reinforcement learning framework with RL agents and environments.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),  # Find packages in the 'src' directory
    package_dir={"": "src"},  # The root package is in the 'src' folder
    install_requires=[
        "jax",
        "gymnax==0.0.8",
        "tyro",
        "flax",
        "chex",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
