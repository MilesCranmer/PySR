import setuptools

try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""

exec(open("pysr/version.py").read())

setuptools.setup(
    name="pysr",
    version=__version__,
    author="Miles Cranmer",
    author_email="miles.cranmer@gmail.com",
    description="Simple and efficient symbolic regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MilesCranmer/pysr",
    # Read from requirements.txt:
    install_requires=open("requirements.txt").read().splitlines(),
    packages=setuptools.find_packages(),
    package_data={"pysr": ["../Project.toml", "../datasets/*"]},
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
