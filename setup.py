import importlib.util
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

extra_installs = []

torch_installed = (importlib.util.find_spec('torch') is not None)
install_sympytorch = torch_installed

if install_sympytorch:
    extra_installs.append('sympytorch')

setuptools.setup(
    name="pysr", # Replace with your own username
    version="0.6.0rc2",
    author="Miles Cranmer",
    author_email="miles.cranmer@gmail.com",
    description="Simple and efficient symbolic regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MilesCranmer/pysr",
    install_requires=([
        "numpy",
        "pandas",
        "sympy"
        ] + extra_installs),
    packages=setuptools.find_packages(),
    package_data={
        'pysr': ['../Project.toml', '../datasets/*']
    },
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
