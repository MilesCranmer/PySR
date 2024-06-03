import os

from setuptools import setup

if os.path.exists(".git"):
    kwargs = {
        "use_scm_version": {
            "write_to": "pysr/version.py",
        },
        "setup_requires": ["setuptools", "setuptools_scm"],
    }
else:
    # Read from pyproject.toml directly
    import re

    with open(os.path.join(os.path.dirname(__file__), "pyproject.toml")) as f:
        data = f.read()
        # Find the version
        version = re.search(r'version = "(.*)"', data).group(1)

    # Write the version to version.py
    with open(os.path.join(os.path.dirname(__file__), "pysr", "version.py"), "w") as f:
        f.write(f'__version__ = "{version}"')

    kwargs = {
        "use_scm_version": False,
        "version": version,
    }


# Build options are managed in pyproject.toml
setup(**kwargs)
