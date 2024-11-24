import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    require = [x.strip() for x in f.readlines() if not x.startswith("git+")]

setuptools.setup(
    name="paper-ephys-atlas",
    version="0.0.0",
    author="IBL",
    description="paper-ephys-atlas code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/int-brain-lab/paper-ephys-atlas",
    project_urls={
        "Bug Tracker": "https://github.com/int-brain-lab/paper-ephys-atlas/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=require,
    package_dir={"": "sources"},
    packages=setuptools.find_packages(where="sources"),
    # package_data={'easyqc': ['easyqc.ui', 'easyqc.svg']},
    python_requires=">=3.10",
)
