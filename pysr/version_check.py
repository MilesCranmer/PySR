"""Code for checking whether the PySR version is up-to-date"""
import subprocess
import sys
from typing import List
from .version import __version__


def run_pip_command(*commands: List[str]):
    """Run a pip command and return the output

    Parameters
    ----------
    commands : list[str]
        List of commands to run. For example,
        `["install", "pysr"]` will run `pip install pysr`.

    Returns
    -------
    output : str
        Output of the command.
    """
    raw_output = subprocess.run(
        [sys.executable, "-m", "pip", *commands],
        capture_output=True,
        text=True,
    )
    return str(raw_output)


def package_version_check(name: str):
    """Checks if a particular Python package is up-to-date.

    Credit to
    https://stackoverflow.com/questions/58648739/how-to-check-if-python-package-is-latest-version-programmatically

    Parameters
    ----------
    name : str
        Name of the package to check.

    Returns
    -------
    flag : bool
        True if the package is up-to-date, False otherwise.
    current_version : str
        The most up-to-date version of the package.
    latest_version : str
        The currently installed version.
    """

    # TODO: What if someone does not have pip?
    latest_version = run_pip_command("install", f"{name}==random")
    latest_version = latest_version[latest_version.find("(from versions:") + 15 :]
    latest_version = latest_version[: latest_version.find(")")]
    latest_version = latest_version.replace(" ", "").split(",")[-1]

    install_check_failed = latest_version == "none"
    if install_check_failed:
        # Could be due to no internet, or other reason.
        return True, "none", "none"

    if name == "pysr":
        current_version = __version__
    else: 
        current_version = run_pip_command("show", name)
        current_version = current_version[current_version.find("Version:") + 8 :]
        current_version = current_version[: current_version.find("\\n")].replace(" ", "")

    if latest_version == current_version:
        return True, current_version, latest_version
    else:
        return False, current_version, latest_version


def pysr_version_check():
    """Checks if PySR is up-to-date"""
    return package_version_check("pysr")
