import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pysr", # Replace with your own username
    version="0.3.21",
    author="Miles Cranmer",
    author_email="miles.cranmer@gmail.com",
    description="Simple and efficient symbolic regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MilesCranmer/pysr",
    install_requires=[
        "numpy",
        "pandas",
        "sympy"
        ],
    packages=setuptools.find_packages(),
    package_data={
        'pysr': ['../julia/*.jl']
    },
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.3',
)
